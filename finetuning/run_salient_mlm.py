# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""


import torch
import logging
import os
import time
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
from filelock import FileLock

from torch.utils.data import ConcatDataset
from my_trainer import EarlyStopTrainer as Trainer

from salient_collator import SalientMaskDataCollator

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
    set_seed,
)


import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field()
    control_group: str = field(default="baseline")
    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
            "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    salient_mode: str = field(default="sentiment")


@dataclass
class CustomArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    patience: int = field(default=500)
    resume: Optional[bool] = field(default=False)


class LineByLineTextDatasetGUID(object):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_dir,
        model_name,
        file_path,
        block_size,
        mode,
        salient_mode,
        control_group,
    ):

        cached_features_file = os.path.join(
            data_dir,
            "lm_cached_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                block_size,
            ),
        )
        lock_path = cached_features_file + ".lock"

        with FileLock(lock_path):

            if os.path.isfile(cached_features_file):
                self.examples = torch.load(cached_features_file)
            else:
                start = time.time()
                with open(file_path, encoding="utf-8") as f:
                    lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

                batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
                self.examples = batch_encoding["input_ids"]
                torch.save(self.examples, cached_features_file)
                cache_time = time.time() - start
                print(f"caching the features took {cache_time:.2f}s.")

        self.mode = mode
        self.salient_mode = salient_mode
        self.tokenizer = tokenizer
        self.control_group = control_group

        model_name = model_name.replace("-", "_")
        salient_masks = torch.load(
            os.path.join(data_dir, f"cached_{mode}_{model_name}_{block_size}.{salient_mode}_mask")
        ).bool()

        filter = salient_masks.any(dim=1)
        mask_pct = (filter.sum().float() / len(filter)).item()
        print(f"{mask_pct:.2%} lines have available salient masks.")
        self.examples = [self.examples[i.item()] for i in filter.nonzero(as_tuple=False)]
        self.salient_masks = salient_masks[filter]

    def __getitem__(self, i):
        return self.examples[i], self.salient_masks[i]

    def __len__(self):
        return len(self.examples)


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    model_name,
    evaluate: bool = False,
):
    if os.path.isdir(model_name):
        # HACK: need to fix this later
        model_name = "bert-base-uncased"

    def _dataset(file_path):
        return LineByLineTextDatasetGUID(
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            model_name=model_name,
            file_path=file_path,
            block_size=args.block_size,
            mode="dev" if evaluate else "train",
            salient_mode=args.salient_mode,
            control_group=args.control_group,
        )

    if evaluate:
        return _dataset(args.eval_data_file)
    elif args.train_data_files:
        return ConcatDataset([_dataset(f) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    train_dataset = get_dataset(
        data_args,
        model_name=model_args.model_name_or_path,
        tokenizer=tokenizer,
    )
    eval_dataset = get_dataset(
        data_args,
        model_name=model_args.model_name_or_path,
        tokenizer=tokenizer,
        evaluate=True,
    )

    data_collator = SalientMaskDataCollator(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        control_group=data_args.control_group,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
        patience=custom_args.patience,
        resume=custom_args.resume,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        model, train_output = trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        for control in ["positive", "negative"]:
            data_collator = SalientMaskDataCollator(
                tokenizer=tokenizer,
                mlm_probability=data_args.mlm_probability,
                control_group=control,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=eval_dataset,
                eval_dataset=eval_dataset,
                prediction_loss_only=True,
                patience=custom_args.patience,
                resume=custom_args.resume,
            )

            eval_output = trainer.evaluate()[0]

            results[f"{control}_eval_loss"] = eval_output["eval_loss"]

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm_diff_control.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(results.keys()):
                    logger.info("  %s = %s", key, str(results[key]))
                    writer.write("%s = %s\n" % (key, str(results[key])))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
