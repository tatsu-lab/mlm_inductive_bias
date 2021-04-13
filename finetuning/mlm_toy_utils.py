import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from transformers import BertForSequenceClassification
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.modeling_outputs import SequenceClassifierOutput


class BertForFinetuning(BertForSequenceClassification):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_token_idx = input_ids.ne(self.config.pad_token_id).long().sum(dim=-1) - 2  # 0-index and [SEP] token
        masked_output = outputs[0][
            torch.arange(last_token_idx.size(0)),
            last_token_idx
        ]

        temp_output = self.dropout(masked_output)
        if self.project is not None:
            temp_output = self.project(temp_output)
        logits = self.classifier(temp_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@dataclass
class DataCollatorForPretraining(DataCollatorForLanguageModeling):
    cloze_pct: float = 1.

    def __call__(self, examples):
        examples = [e.input_ids for e in examples]
        return super(DataCollatorForPretraining, self).__call__(examples)

    def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ):
        labels = torch.full_like(inputs, fill_value=-100)

        last_token_idx = inputs.ne(self.tokenizer.pad_token_id).long().sum(dim=-1) - 2  # 0-index and [SEP] token
        lengths = inputs.ne(self.tokenizer.pad_token_id).long().sum(
            dim=-1) - 1  # 0-index and [SEP] token
        mask_token_idx = [torch.randint(1, lengths[i]-1, (1,)) for i in range(lengths.size(0))] # don't mask [CLS] token and the last token
        mask_token_idx = torch.tensor(mask_token_idx)
        cloze_mask = torch.bernoulli(torch.full(last_token_idx.size(), fill_value=self.cloze_pct)).bool()
        mask_token_idx[cloze_mask] = last_token_idx[cloze_mask]

        labels[torch.arange(inputs.size(0)), mask_token_idx] = inputs[torch.arange(inputs.size(0)), mask_token_idx]
        inputs[torch.arange(inputs.size(0)), mask_token_idx] = self.tokenizer.mask_token_id

        return inputs, labels

@dataclass
class DataCollatorForFinetuning(DataCollatorForLanguageModeling):

    def __call__(self, examples):
        labels = torch.tensor([e.label for e in examples], dtype=torch.long)
        examples = [e.input_ids for e in examples]
        batch = super(DataCollatorForFinetuning, self).__call__(examples)
        batch["labels"] = labels
        return batch

    def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ):
        labels = torch.full_like(inputs, fill_value=-100)
        last_token_idx = inputs.ne(self.tokenizer.pad_token_id).long().sum(dim=-1) - 2 # 0-index and [SEP] token
        labels[torch.arange(inputs.size(0)), last_token_idx] = inputs[torch.arange(inputs.size(0)), last_token_idx]
        inputs[torch.arange(inputs.size(0)), last_token_idx] = self.tokenizer.mask_token_id
        return inputs, labels
