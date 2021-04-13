"""
This script computes word masks based on sentiment lexicons
"""
import os
import torch
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import GlueDataset as Dataset


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="./data/SST-2", help="path to the dir containing lm data.")
parser.add_argument("--lexicon-dir", type=str, default="./data/sentiment_lexicon", help="path to the dir containing sentiment lexicon.")
parser.add_argument("--tokenizer-name", type=str, default="bert-base-uncased", help="name of the tokenizer to use.")
parser.add_argument("--block_size", type=int, default=72, help="maximum length of the mask")
args = parser.parse_args()

positive_words = set()
with open(os.path.join(args.lexicon_dir, "positive-words.txt"), "r", encoding="ISO-8859-1") as f:
    for line in f:
        line = line.strip()
        # skip the initial comments with ; and empty lines
        if not line.startswith(";") and len(line) > 0:
            positive_words.add(line.lower())

negative_words = set()
with open(os.path.join(args.lexicon_dir, "negative-words.txt"), "r", encoding="ISO-8859-1") as f:
    for line in f:
        line = line.strip()
        # skip the initial comments with ; and empty lines
        if not line.startswith(";") and len(line) > 0:
            negative_words.add(line.lower())

salient_words = positive_words | negative_words

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

splits = ["train", "dev", "test"]

for split in splits:
    with open(os.path.join(args.data_dir, f"{split}.lm"), "r") as f:
        all_sens = [s.strip() for s in f.readlines()]

    salient_word_masks = torch.zeros(len(all_sens), args.block_size, dtype=torch.bool)

    total_word_count = 0
    salient_word_count = 0

    # Main loop that handles subword tokenization
    for i, sen in tqdm(enumerate(all_sens), total=len(all_sens)):
        words = sen.split()
        curr_idx = 1  # skip the [CLS] token
        total_word_count += len(words)
        for word in words:
            tokens = tokenizer.tokenize(word)

            # Need to truncate SQuAD
            if curr_idx + len(tokens) > args.block_size:
                raise ValueError("Encountered examples longer than block size.")

            if word in salient_words:
                salient_word_count += 1
                for j in range(len(tokens)):
                    salient_word_masks[i, curr_idx + j] = 1
            curr_idx += len(tokens)

    print(f"{(salient_word_count/total_word_count):.2%} salient words")
    salient_pct = salient_word_masks.any(dim=1).sum().float() / len(all_sens)
    print(f"{split} {salient_pct:.2%} documents have salient words")

    torch.save(
        salient_word_masks,
        os.path.join(
            args.data_dir,
            f"cached_{split}_{args.tokenizer_name.replace('-', '_')}_{args.block_size}.sentiment_mask",
        ),
    )
