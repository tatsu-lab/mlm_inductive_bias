"""
Simple script for compute the mean and standard error accross different random seeds.
"""
from pathlib import Path
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser(description="Aggregating Results")
parser.add_argument("--dataset", default="sst-2", choices=["sst-2", "agnews", "hyperpartisan"])
parser.add_argument("--seed_start", type=int, default=1)
parser.add_argument("--seed_end", type=int, default=6)
args = parser.parse_args()

aggregate_data = {}

for setting in ["baseline", "positive", "negative"]:
    per_setting_data = defaultdict(list)
    for seed in range(args.seed_start, args.seed_end):

        file_path = Path(
            f"out/bert_finetune/{args.dataset}/{setting}/seed_{seed}/test_results.txt"
        )
        with open(file_path, "r") as f:
            lines = f.readlines()
            acc = float(lines[1].split("=")[1])
        per_setting_data["acc"].append(acc)

    aggregate_data[setting] = per_setting_data

settings = ["negative", "baseline", "positive"]
sst_means = dict()
sst_confs = dict()

for setting in settings:
    mean = np.mean(aggregate_data[setting]["acc"])
    conf = np.std(aggregate_data[setting]["acc"]) / np.sqrt(len(aggregate_data[setting]["acc"])) * 1.96
    sst_means[setting] = mean
    sst_confs[setting] = conf

for key in sst_means:
    print(f"setting: {key}, accuracy: {sst_means[key]:.2%}+-{sst_confs[key]:.2%}")
