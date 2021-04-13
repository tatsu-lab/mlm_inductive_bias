import torch
import os
from tqdm import trange
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
    set_seed,
)
from torch.nn.utils.rnn import pad_sequence

import numpy as np


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
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field()
    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    batch_size: int = field(default=512)
    mc_samples: int = field(default=5)
    burn_in: int = field(default=2)


class LineByLineTextDataset(object):
    def __init__(self, tokenizer, file_path):

        with open(file_path, encoding="utf-8") as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]

        batch_encoding = tokenizer(
            lines, add_special_tokens=True,
        )

        self.examples = batch_encoding

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
):

    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.train_data_file,
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
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

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )

    model = AutoModelWithLMHead.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()

    train_dataset = get_dataset(
        data_args,
        tokenizer=tokenizer,
    )

    if training_args.debug:
        for k, v in train_dataset.examples.items():
            train_dataset.examples[k] = v[:10]

    # sort by lens to do clever batching
    lens = torch.tensor([len(sen) for sen in train_dataset.examples["input_ids"]])
    sorted_idx = lens.sort(dim=0, descending=True)[1].numpy().tolist()

    for k, v in train_dataset.examples.items():
        train_dataset.examples[k] = [v[idx] for idx in sorted_idx]

    def unique_pairs(n):
        """Produce pairs of indexes in range(n)"""
        for i in range(n):
            for j in range(i + 1, min(n, i + 10)):
                yield i, j

    all_pairs = []
    all_input_ids = []
    all_attention_mask = []
    all_type_ids = []

    for i, sen in enumerate(train_dataset.examples["input_ids"]):
        pairs = torch.tensor(list(unique_pairs(len(sen) - 2)), dtype=torch.long)
        temp_sen = (
            torch.tensor(sen, dtype=torch.long, device="cuda")
            .unsqueeze(0)
            .expand(pairs.size(0), len(sen))
        )
        all_pairs.append(pairs)
        for j in range(pairs.size(0)):
            all_input_ids.append(temp_sen[j])
            all_attention_mask.append(
                torch.tensor(
                    train_dataset.examples["attention_mask"][i], dtype=torch.bool
                )
            )
            all_type_ids.append(
                torch.tensor(
                    train_dataset.examples["token_type_ids"][i], dtype=torch.long
                )
            )

    all_pairs = torch.cat(all_pairs, dim=0)

    # create data cache for logits first
    mi_pairs = [[], []]

    chain_samples = [
        np.zeros(
            (
                data_args.batch_size,
                data_args.mc_samples - data_args.burn_in,
            ),
            dtype=int,
        ),
        np.zeros(
            (
                data_args.batch_size,
                data_args.mc_samples - data_args.burn_in,
            ),
            dtype=int,
        ),
    ]
    chain_logits = [
        np.zeros(
            (
                data_args.batch_size,
                data_args.mc_samples - data_args.burn_in,
                len(tokenizer),
            ),
            dtype="float32",
        ),
        np.zeros(
            (
                data_args.batch_size,
                data_args.mc_samples - data_args.burn_in,
                len(tokenizer),
            ),
            dtype="float32",
        ),
    ]

    # Wrapping around model forward pass
    def forward_lm(inputs):
        model.eval()
        with torch.no_grad():
            activations = model(
                **inputs
                # sen, attention_mask=torch.ones_like(temp_sen), token_type_ids=torch.zeros_like(temp_sen)
            )[0]
            logits = activations.log_softmax(dim=-1)
        return logits

    for batch_start in trange(0, all_pairs.size(0), data_args.batch_size):
        batch_pairs = (
            all_pairs[batch_start : batch_start + data_args.batch_size].to("cuda") + 1
        )  # offset by [CLS]
        batch_pairs = batch_pairs.cuda()
        batch_input_ids, batch_attention_mask, batch_type_ids = (
            all_input_ids[batch_start : batch_start + data_args.batch_size],
            all_attention_mask[batch_start : batch_start + data_args.batch_size],
            all_type_ids[batch_start : batch_start + data_args.batch_size],
        )

        batch_input_ids = pad_sequence(
            batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        batch_attention_mask = pad_sequence(
            batch_attention_mask, batch_first=True, padding_value=0
        )
        batch_type_ids = pad_sequence(batch_type_ids, batch_first=True, padding_value=0)
        inputs = {
            "input_ids": batch_input_ids.cuda(),
            "attention_mask": batch_attention_mask.cuda(),
            "token_type_ids": batch_type_ids.cuda(),
        }

        if len(batch_pairs) < data_args.batch_size:
            chain_samples = [
                np.zeros(
                    (
                        len(batch_pairs),
                        data_args.mc_samples - data_args.burn_in,
                    ),
                    dtype=int,
                ),
                np.zeros(
                    (
                        len(batch_pairs),
                        data_args.mc_samples - data_args.burn_in,
                    ),
                    dtype=int,
                ),
            ]
            chain_logits = [
                np.zeros(
                    (
                        len(batch_pairs),
                        data_args.mc_samples - data_args.burn_in,
                        len(tokenizer),
                    ),
                    dtype="float32",
                ),
                np.zeros(
                    (
                        len(batch_pairs),
                        data_args.mc_samples - data_args.burn_in,
                        len(tokenizer),
                    ),
                    dtype="float32",
                ),
            ]

        inputs["input_ids"] = inputs["input_ids"].scatter(
            dim=1, index=batch_pairs, value=tokenizer.mask_token_id
        )

        for n in trange(data_args.mc_samples, leave=False):
            for idx_n in range(2):
                inputs["input_ids"] = inputs["input_ids"].scatter(
                    dim=1,
                    index=batch_pairs[:, idx_n : idx_n + 1],
                    value=tokenizer.mask_token_id,
                )
                logits = forward_lm(inputs)
                probs = logits[
                    torch.arange(logits.size(0)), batch_pairs[:, idx_n]
                ].exp()
                sample = torch.multinomial(probs, 1)
                if n >= data_args.burn_in:
                    chain_samples[idx_n][:, n - data_args.burn_in] = (
                        sample.squeeze(-1).cpu().numpy()
                    )
                    chain_logits[idx_n][:, n - data_args.burn_in] = probs.cpu().numpy()

                inputs["input_ids"] = inputs["input_ids"].scatter(
                    dim=1,
                    index=batch_pairs[:, idx_n : idx_n + 1],
                    src=sample.squeeze(0),
                )

        for idx_n in range(2):
            i_j_xr = (
                np.take_along_axis(
                    chain_logits[idx_n], np.expand_dims(chain_samples[idx_n], 2), axis=2
                )
            ).squeeze(2)

            i_xr = np.take_along_axis(
                chain_logits[idx_n],
                np.repeat(
                    np.expand_dims(chain_samples[idx_n], 1),
                    chain_samples[idx_n].shape[1],
                    axis=1,
                ),
                axis=2,
            )
            i_xr = i_xr.mean(axis=1)
            mi = (np.log(i_j_xr) - np.log(i_xr)).mean(axis=1)
            mi_pairs[idx_n].append(torch.from_numpy(mi))

    all_log_mi = []
    mi_pairs[0], mi_pairs[1] = torch.cat(mi_pairs[0], dim=0), torch.cat(
        mi_pairs[1], dim=0
    )
    curr = 0
    for sen in train_dataset.examples["input_ids"]:
        mi_matrix = torch.zeros(len(sen) - 2, len(sen) - 2)
        temp_pairs = list(unique_pairs(len(sen) - 2))
        for n, (i, j) in enumerate(temp_pairs):
            mi_matrix[i, j] = mi_pairs[0][curr + n]
            mi_matrix[j, i] = mi_pairs[1][curr + n]
        all_log_mi.append(mi_matrix)
        curr += len(temp_pairs)

    sort_back_dict = {b: a for a, b in enumerate(sorted_idx)}
    unsorted_log_mi = []
    for i in range(len(all_log_mi)):
        unsorted_log_mi.append(all_log_mi[sort_back_dict[i]])

    os.makedirs(training_args.output_dir, exist_ok=True)
    torch.save(
        unsorted_log_mi,
        os.path.join(training_args.output_dir, "all_log_mi.pt"),
    )


if __name__ == "__main__":
    main()
