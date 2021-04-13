from typing import Dict, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.tokenization_utils_base import BatchEncoding


class SalientMaskDataCollator:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    - change masking behavior based on the salient masks and different control group
    """

    def __init__(self, tokenizer, mlm_probability, control_group="baseline"):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.control_group = control_group

    def __call__(self, examples) -> Dict[str, torch.Tensor]:

        examples, salient_masks = list(zip(*examples))

        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        salient_masks = self._tensorize_batch(salient_masks)
        salient_masks = salient_masks[:, : batch.size(1)]

        inputs, labels = self.mask_tokens(batch, salient_masks)
        return {"input_ids": inputs, "labels": labels}

    def _tensorize_batch(self, examples) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(
                examples, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

    def mask_tokens(
        self, inputs: torch.Tensor, salient_masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        if self.control_group != "baseline":
            # make sure that there are roughly same amount of masks in different control groups
            old_row_sum = probability_matrix.sum(dim=1).float()

            if self.control_group == "positive":
                probability_matrix.masked_fill_(~salient_masks, value=0.0)
            elif self.control_group == "negative":
                probability_matrix.masked_fill_(salient_masks, value=0.0)
            else:
                raise ValueError(f"Invalid Control Group Option, {self.control_group}")

            new_sample_prob = (
                old_row_sum / probability_matrix.sum(dim=1).float()
            ) * self.mlm_probability
            new_sample_prob.clamp_(0, 1)

            probability_matrix = probability_matrix.ne(
                0
            ).float() * new_sample_prob.unsqueeze(dim=1)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
