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
""" SuperGLUE processors and helpers """


import logging
from collections import defaultdict

import numpy as np

from transformers.file_utils import is_tf_available
from transformers import glue_output_modes, glue_tasks_num_labels, glue_processors
from hyperpartisan_utils import *
from agnews_utils import *


import torch
import time
from enum import Enum
from typing import List, Optional, Union
from filelock import FileLock

from torch.utils.data.dataset import Dataset
from transformers import GlueDataTrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Adapted from glue_convert_examples_to_features

    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: SuperGLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            NB(AW): Writing predictions assumes the labels are in the same order as when building features.
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        # Legacy code
        # if isinstance(example, SpanClassificationExample):
        #     inputs_a, span_locs_a = tokenize_tracking_span(tokenizer, example.text_a, example.spans_a)
        #     if example.spans_b is not None:
        #         inputs_b, span_locs_b = tokenize_tracking_span(tokenizer, example.text_b, example.spans_b)
        #         num_non_special_tokens = len(inputs_a["input_ids"]) + len(inputs_b["input_ids"]) - 4

        #         # TODO(AW): assumption is same number of non-special tokens + sos + eos
        #         #   This handles varying number of intervening tokens (e.g. different models)
        #         inputs = tokenizer.encode_plus(
        #             example.text_a,
        #             example.text_b,
        #             add_special_tokens=True,
        #             max_length=max_length,
        #             return_token_type_ids=True,
        #         )
        #         num_joiner_specials = len(inputs["input_ids"]) - num_non_special_tokens - 2
        #         offset = len(inputs_a["input_ids"]) - 1 + num_joiner_specials - 1
        #         span_locs_b = [(s + offset, e + offset) for s, e in span_locs_b]
        #         span_locs = span_locs_a + span_locs_b
        #         input_ids = inputs["input_ids"]
        #         token_type_ids = inputs["token_type_ids"]

        #         if num_joiner_specials == 1:
        #             tmp = inputs_a["input_ids"] + inputs_b["input_ids"][1:]
        #         elif num_joiner_specials == 2:
        #             tmp = inputs_a["input_ids"] + inputs_b["input_ids"]
        #         else:
        #             assert False, "Something is wrong"

        #         # check that the length of the input ids is expected (not necessarily the exact ids)
        #         assert len(input_ids) == len(tmp), "Span tracking tokenization produced inconsistent result!"

        #     else:
        #         input_ids, token_type_ids = inputs_a["input_ids"], inputs_a["token_type_ids"]
        #         span_locs = span_locs_a

        # else:
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            # TODO(AW): will mess up span tracking
            assert False, "Not implemented correctly wrt span tracking!"
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids

        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        if output_mode in ["classification", "span_classification"]:
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input text: %s" % tokenizer.decode(input_ids, clean_up_tokenization_spaces=False))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        # if isinstance(example, SpanClassificationExample):
        #     feats = SpanClassificationFeatures(
        #         guid=example.guid,
        #         input_ids=input_ids,
        #         span_locs=span_locs,
        #         attention_mask=attention_mask,
        #         token_type_ids=token_type_ids,
        #         label=label,
        #     )
        # else:
        feats = InputFeatures(
            # guid=example.guid,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label=label,
        )

        features.append(feats)

    if is_tf_available() and is_tf_dataset:
        # TODO(AW): include span classification version

        def gen():
            for ex in features:
                yield (
                    {
                        "guid": ex.guid,
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "guid": tf.TensorShape([None]),
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features

glue_processors["hyperpartisan"] = HyperPartisanProcessor
glue_tasks_num_labels["hyperpartisan"] = hyperpartisan_num_labels
glue_output_modes["hyperpartisan"] = hyperpartisan_output_mode
glue_processors["agnews"] = AGNewsProcessor
glue_tasks_num_labels["agnews"] = agnews_num_labels
glue_output_modes["agnews"] = agnews_output_mode


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class CustomDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        # if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
        #     RobertaTokenizer,
        #     RobertaTokenizerFast,
        #     XLMRobertaTokenizer,
        #     BartTokenizer,
        #     BartTokenizerFast,
        # ):
        #     # HACK(label indices are swapped in RoBERTa pretrained model)
        #     label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

    def downsample(self, data_size_per_class):
        class_idx = defaultdict(list)
        for i, feat in enumerate(self.features):
            class_idx[feat.label].append(i)

        selected_idx = []
        rng = np.random.default_rng(0)  # fix the seed for generating training set
        for class_label in class_idx.keys():
            rng.shuffle(class_idx[class_label])
            selected_idx += class_idx[class_label][:data_size_per_class]

        self.features = [self.features[i] for i in selected_idx]
