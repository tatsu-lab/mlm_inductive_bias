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
""" Finetuning the library models for sequence classification on GLUE."""


from transformers.data.processors.glue import glue_convert_examples_to_features
from copy import deepcopy

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch
import numpy as np
from collections import defaultdict

from transformers import BertConfig, BertForMaskedLM
from transformers import AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

from my_trainer import EarlyStopTrainer as Trainer
from mlm_toy_utils import BertForFinetuning, DataCollatorForFinetuning, DataCollatorForPretraining


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
    overwrite_pretrain_dir: bool = field(
        default=False,
    )
    nhid: int = field(
        default=64,
    )
    nlayers: int = field(
        default=2,
    )

def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return glue_compute_metrics(task_name, preds, p.label_ids)

    return compute_metrics_fn

def build_mlm_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        mask = labels.ne(-100)
        acc = (preds[mask]).eq(labels[mask]).float().mean().item()
        cloze_mask = (labels.eq(4366)) | (labels.eq(3112))
        cloze_acc = (preds[cloze_mask]).eq(labels[cloze_mask]).float().mean().item()
        return {"acc": acc, "cloze_acc": cloze_acc}

    return compute_metrics_fn

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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


    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    # Get datasets
    pretrain_dataset = GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir)
    finetune_dataset = GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir)
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)

    # monkey patch to add prompt
    for i, dataset in enumerate([pretrain_dataset, finetune_dataset, eval_dataset]):
        if i < 2:
            dataset.examples = dataset.processor.get_train_examples(data_args.data_dir)
        else:
            dataset.examples = dataset.processor.get_dev_examples(data_args.data_dir)

        for example in dataset.examples:
            example.text_a += " positive" if example.label == "1" else " negative"

        dataset.features = glue_convert_examples_to_features(
            dataset.examples,
            tokenizer,
            max_length=data_args.max_seq_length,
            label_list=dataset.label_list,
            output_mode=dataset.output_mode,
        )

    pretrain_dataset.features = pretrain_dataset.features[: len(pretrain_dataset) // 2]
    finetune_dataset.features = finetune_dataset.features[len(pretrain_dataset) // 2 :]

    # TODO: pass in args for these
    config = BertConfig.from_dict(
        {
            "num_hidden_layers": model_args.nlayers,
            "num_attention_heads": 1,
            "hidden_size": model_args.nhid,
            "intermediate_size": model_args.nhid,
            # "tie_word_embeddings": False
        }
    )

    def load_or_train(cloze_pct):
        model = BertForMaskedLM(config).cuda()
        collator = DataCollatorForPretraining(tokenizer=tokenizer, mlm=True, cloze_pct=cloze_pct)
        overwrite = model_args.overwrite_pretrain_dir
        log = f"{cloze_pct:.0%} Cloze Pre-trained Model"
        output_dir = f"cloze_{cloze_pct:.0%}"
        model_path = os.path.join(training_args.output_dir, output_dir, "checkpoint-best")

        if os.path.isfile(os.path.join(model_path, "pytorch_model.bin")) and not overwrite:
            model = model.from_pretrained(model_path)
            logger.info(f"Loaded {log}")

        default_output_dir = training_args.output_dir
        training_args.output_dir = os.path.join(default_output_dir, output_dir)

        # Set seed
        set_seed(training_args.seed)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=pretrain_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            patience=-1,
            task="mlm",
            early_stop_metric="acc"
        )
        trainer.compute_metrics = build_mlm_compute_metrics_fn(eval_dataset.args.task_name)
        logger.info(f"Evaluating {cloze_pct:.0%} Cloze Model.")
        trainer.evaluate()

        if not os.path.isfile(os.path.join(model_path, "pytorch_model.bin")) or overwrite:
            logger.info(f"Training {log}")
            trainer.train()
            trainer.save_model()

        model = model.from_pretrained(model_path)
        training_args.output_dir = default_output_dir

        return model

    settings = np.linspace(0, 1, 6).tolist()[::-1]
    models = {}
    for pct in settings:
        models[pct] = load_or_train(cloze_pct=pct)

    settings = ["scratch"] + settings

    for setting in settings:
        if setting == "scratch":
            setting_name = setting
        else:
            setting_name = f"cloze_{setting:.0%}"
        out_dir = os.path.join(training_args.output_dir, setting_name)
        if os.path.isfile(os.path.join(out_dir, "finetune_results.pt")):
            print(f"Finetuning for {setting_name} has been completed.")
            continue
        else:
            print(f"Start Finetuning for {setting_name}")
        accs = {}
        losses = {}
        for data_size in [10, 30, 100, 300, 1000, 3000, 10000]:
            acc_across_seeds = []
            loss_across_seeds = []
            for subsample_seed in range(4):
                set_seed(subsample_seed)
                finetune_config = deepcopy(config)
                finetune_config.num_labels = num_labels
                finetune_model = BertForFinetuning(finetune_config).cuda()
                collator = DataCollatorForFinetuning(tokenizer=tokenizer, mlm=True)
                if setting != "scratch":
                    with torch.no_grad():
                        finetune_model.bert = deepcopy(models[setting].bert)
                        finetune_model.project = deepcopy(models[setting].cls.predictions.transform)
                        finetune_model.classifier.weight[0] = models[setting].cls.predictions.decoder.weight[4366]
                        finetune_model.classifier.weight[1] = models[setting].cls.predictions.decoder.weight[3112]
                        finetune_model.lm_head = models[setting].cls.predictions
                        finetune_model.classifier.bias.weight = models[setting].cls.predictions.decoder.bias[[4366, 3112]]
                else:
                    finetune_model.project = None

                rand_idx = torch.randperm(len(finetune_dataset))[:data_size]
                temp_dataset = deepcopy(finetune_dataset)
                temp_dataset.features = [temp_dataset.features[i] for i in list(rand_idx)]

                training_args.logging_steps = 0
                training_args.eval_steps = 100
                training_args.learning_rate = 1e-4
                if data_size * training_args.num_train_epochs < 1000 * training_args.per_device_train_batch_size:
                    training_args.max_steps = 1000
                else:
                    training_args.max_steps = data_size * training_args.num_train_epochs // training_args.per_device_train_batch_size

                finetune_trainer = Trainer(
                    model=finetune_model,
                    args=training_args,
                    train_dataset=temp_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=collator,
                    early_stop_metric="acc",
                    patience=-1,
                )

                finetune_trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
                finetune_trainer.train()
                finetune_trainer.model = finetune_trainer.best_model
                eval_result, _ = finetune_trainer.evaluate(eval_dataset=eval_dataset)
                acc_across_seeds.append(eval_result["eval_acc"])
                loss_across_seeds.append(eval_result["eval_loss"])

            acc_mean, acc_std = np.mean(acc_across_seeds), np.std(acc_across_seeds)
            loss_mean, loss_std = np.mean(loss_across_seeds), np.std(loss_across_seeds)
            accs[data_size] = (acc_mean, acc_std)
            losses[data_size] = (loss_mean, loss_std)
            print(f"Setting: {setting} Acc: {acc_mean}+-{acc_std} Loss: {loss_mean}+-{loss_std}")

        results = {"accs": accs, "losses": losses}
        os.makedirs(out_dir, exist_ok=True)
        torch.save(results, os.path.join(out_dir, "finetune_results.pt"))


if __name__ == "__main__":
    main()
