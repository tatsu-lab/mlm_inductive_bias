"""
This script converts the tsv format of classification data (like SST-2) into txt format of LM data
"""
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_dir", default="./data/SST-2", help="path to the directory containing data")

args = parser.parse_args()

for split in ["train", "test", "dev"]:
    original_data = pd.read_csv(f"{args.data_dir}/{split}.tsv", delimiter="\t")
    max_len = max([len(s.split(" ")) for s in original_data["sentence"]])
    with open(f"{args.data_dir}/{split}.lm", "w") as f:
        for sen in original_data["sentence"]:
            f.write(sen+"\n")
