"""
Adapted from https://github.com/john-hewitt/structural-probes
"""
import numpy as np
from collections import namedtuple, defaultdict
from argparse import ArgumentParser
import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import math

argp = ArgumentParser()
argp.add_argument("--output_dir", default="./out/eval_results")
argp.add_argument("--raw_data_dir", default="./data/")
argp.add_argument("--model_name", default="bert-base-cased")
argp.add_argument("--split", default="test")
argp.add_argument("--print_tikz", action="store_true",default=False)
argp.add_argument("--pmi_clamp_zero", action="store_true",default=False)
argp.add_argument("--word_piece_agg_space", type=str, default="log", choices=["log", "exp"])
argp.add_argument("--word_piece_agg_type", type=str, default="avg", choices=["avg", "min", "max"])
argp.add_argument("--no_prefix_space", action="store_true")
argp.add_argument("--word_piece_pmi_path", type=str, default="", required=True)


def unique_pairs(n):
    """Produce pairs of indexes in range(n)"""
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


class Task:

    """Abstract class representing a linguistic task mapping texts to labels."""

    @staticmethod
    def labels(observation):
        """Maps an observation to a matrix of labels.

        Should be overriden in implementing classes.
        """
        raise NotImplementedError


class ParseDistanceTask(Task):
    """Maps observations to dependency parse distances between words."""

    @staticmethod
    def labels(observation):
        """Computes the distances between all pairs of words; returns them as a torch tensor.

        Args:
          observation: a single Observation class for a sentence:
        Returns:
          A torch tensor of shape (sentence_length, sentence_length) of distances
          in the parse tree as specified by the observation annotation.
        """
        sentence_length = len(observation[0])  # All observation fields must be of same length
        distances = torch.zeros((sentence_length, sentence_length))
        for i in range(sentence_length):
            for j in range(i, sentence_length):
                i_j_distance = ParseDistanceTask.distance_between_pairs(observation, i, j)
                distances[i][j] = i_j_distance
                distances[j][i] = i_j_distance
        return distances

    @staticmethod
    def distance_between_pairs(observation, i, j, head_indices=None):
        """Computes path distance between a pair of words

        TODO: It would be (much) more efficient to compute all pairs' distances at once;
              this pair-by-pair method is an artefact of an older design, but
              was unit-tested for correctness...

        Args:
          observation: an Observation namedtuple, with a head_indices field.
              or None, if head_indies != None
          i: one of the two words to compute the distance between.
          j: one of the two words to compute the distance between.
          head_indices: the head indices (according to a dependency parse) of all
              words, or None, if observation != None.

        Returns:
          The integer distance d_path(i,j)
        """
        if i == j:
            return 0
        if observation:
            head_indices = []
            number_of_underscores = 0
            for elt in observation.head_indices:
                if elt == "_":
                    head_indices.append(0)
                    number_of_underscores += 1
                else:
                    head_indices.append(int(elt) + number_of_underscores)
        i_path = [i + 1]
        j_path = [j + 1]
        i_head = i + 1
        j_head = j + 1
        while True:
            if not (i_head == 0 and (i_path == [i + 1] or i_path[-1] == 0)):
                i_head = head_indices[i_head - 1]
                i_path.append(i_head)
            if not (j_head == 0 and (j_path == [j + 1] or j_path[-1] == 0)):
                j_head = head_indices[j_head - 1]
                j_path.append(j_head)
            if i_head in j_path:
                j_path_length = j_path.index(i_head)
                i_path_length = len(i_path) - 1
                break
            elif j_head in i_path:
                i_path_length = i_path.index(j_head)
                j_path_length = len(j_path) - 1
                break
            elif i_head == j_head:
                i_path_length = len(i_path) - 1
                j_path_length = len(j_path) - 1
                break
        total_length = j_path_length + i_path_length
        return total_length


def load_conll_dataset(filepath):
    """Reads in a conllx file; generates Observation objects
    For each sentence in a conllx file, generates a single Observation
    object.

    Args:
      filepath: the filesystem path to the conll dataset

    Returns:
      A list of Observations
    """
    observations = []
    lines = (x for x in open(filepath))
    for buf in generate_lines_for_sent(lines):
        conllx_lines = []
        for line in buf:
            conllx_lines.append(line.strip().split("\t"))
        embeddings = [None for x in range(len(conllx_lines))]
        observation = Observations(*zip(*conllx_lines), embeddings)
        observations.append(observation)
    return observations


def generate_lines_for_sent(lines):
    """Yields batches of lines describing a sentence in conllx.

    Args:
      lines: Each line of a conllx file.
    Yields:
      a list of lines describing a single sentence in conllx.
    """
    buf = []
    for line in lines:
        if line.startswith("#"):
            continue
        if not line.strip():
            if buf:
                yield buf
                buf = []
            else:
                continue
        else:
            buf.append(line.strip())
    if buf:
        yield buf


class ObservationIterator(Dataset):
    """List Container for lists of Observations and labels for them.

    Used as the iterator for a PyTorch dataloader.
    """

    def __init__(self, observations, task):
        self.observations = observations
        self.set_labels(observations, task)

    def set_labels(self, observations, task):
        """Constructs aand stores label for each observation.

        Args:
          observations: A list of observations describing a dataset
          task: a Task object which takes Observations and constructs labels.
        """
        self.labels = []
        for observation in tqdm(observations, desc="[computing labels]"):
            self.labels.append(task.labels(observation))

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.labels[idx]


def prims_matrix_to_edges(matrix, words, poses):
    """
    Constructs a minimum spanning tree from the pairwise weights in matrix;
    returns the edges.

    Never lets punctuation-tagged words be part of the tree.
    """
    pairs_to_distances = {}
    uf = UnionFind(len(matrix))
    for i_index, line in enumerate(matrix):
        for j_index, dist in enumerate(line):
            if poses[i_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
                continue
            if poses[j_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
                continue
            pairs_to_distances[(i_index, j_index)] = dist
    edges = []
    for (i_index, j_index), distance in sorted(pairs_to_distances.items(), key=lambda x: (x[1], -x[0][0])):
        if uf.find(i_index) != uf.find(j_index):
            uf.union(i_index, j_index)
            edges.append((i_index, j_index))
    return edges


def report_uuas_and_tikz(args, prediction_batches, dataset, split_name):
    """Computes the UUAS score for a dataset and writes tikz dependency latex.

    From the true and predicted distances, computes a minimum spanning tree
    of each, and computes the percentage overlap between edges in all
    predicted and gold trees.

    For the first 20 examples (if not the test set) also writes LaTeX to disk
    for visualizing the gold and predicted minimum spanning trees.

    All tokens with punctuation part-of-speech are excluded from the minimum
    spanning trees.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    uspan_total = 0
    uspan_correct = 0
    adjacent_correct = 0
    rand_correct = 0
    total_sents = 0

    per_relation_stats = defaultdict(lambda: [0, 0])
    per_relation_stats_adjacent = defaultdict(lambda: [0, 0])

    uuas_per_sen = []
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in tqdm(
        zip(prediction_batches, dataset), desc="computing uuas"
    ):
        for prediction, label, length, (observation, _) in zip(
            prediction_batch, label_batch, length_batch, observation_batch
        ):
            words = observation.sentence
            poses = observation.xpos_sentence
            length = int(length)
            assert length == len(observation.sentence)
            prediction = prediction[:length, :length]
            label = label[:length, :length].cpu()

            temp_gold_edges = list(
                zip([int(x) - 1 for x in observation.index], [int(x) - 1 for x in observation.head_indices])
            )
            edge_to_relation = dict(zip([tuple(sorted(e)) for e in temp_gold_edges], observation.governance_relations))
            gold_edges = prims_matrix_to_edges(label, words, poses)
            pred_edges = prims_matrix_to_edges(prediction, words, poses)
            rand_edges = prims_matrix_to_edges(np.random.rand(*prediction.shape), words, poses)

            non_punct = (np.array(observation.upos_sentence) != "PUNCT").nonzero()[0]
            adjacent_edges = [(non_punct[i], non_punct[i + 1]) for i in range(len(non_punct) - 1)]

            pred_edges = set([tuple(sorted(e)) for e in pred_edges])
            gold_edges = set([tuple(sorted(e)) for e in gold_edges])
            adjacent_edges = set([tuple(sorted(e)) for e in adjacent_edges])

            if args.print_tikz and total_sents < 100 :
                print_tikz(args.output_dir, pred_edges, gold_edges, edge_to_relation, words, split_name)

            num_correct = 0
            for edge in gold_edges:
                per_relation_stats[edge_to_relation[edge]][0] += 1
                if edge in pred_edges:
                    num_correct += 1
                    per_relation_stats[edge_to_relation[edge]][1] += 1

            num_correct_adjacent = 0
            for edge in gold_edges:
                per_relation_stats_adjacent[edge_to_relation[edge]][0] += 1
                if edge in adjacent_edges:
                    num_correct_adjacent += 1
                    per_relation_stats_adjacent[edge_to_relation[edge]][1] += 1

            num_correct_rand = len(set([tuple(sorted(e)) for e in rand_edges]).intersection(gold_edges))

            # computed error matrix after filtering adjacency edges
            pred_in_adjacent = pred_edges.intersection(adjacent_edges)
            pred_out_adjacent = pred_edges - adjacent_edges
            gold_in_adjacent = gold_edges.intersection(adjacent_edges)
            gold_out_adjacent = gold_edges - adjacent_edges

            uspan_correct += num_correct
            adjacent_correct += num_correct_adjacent
            rand_correct += num_correct_rand
            uspan_total += len(gold_edges)
            total_sents += 1
            uuas_per_sen.append(uspan_correct / uspan_total)

    uuas = uspan_correct / float(uspan_total)
    uuas_adjacent = adjacent_correct / float(uspan_total)
    uuas_rand = rand_correct / float(uspan_total)

    return (
        uuas,
        uuas_per_sen,
        uuas_adjacent,
        uuas_rand,
        per_relation_stats,
        per_relation_stats_adjacent,
    )


class UnionFind:
    """
    Naive UnionFind implementation for (slow) Prim's MST algorithm

    Used to compute minimum spanning trees for distance matrices
    """

    def __init__(self, n):
        self.parents = list(range(n))

    def union(self, i, j):
        if self.find(i) != self.find(j):
            i_parent = self.find(i)
            self.parents[i_parent] = j

    def find(self, i):
        i_parent = i
        while True:
            if i_parent != self.parents[i_parent]:
                i_parent = self.parents[i_parent]
            else:
                break
        return i_parent


def custom_pad(batch_observations):
    """Pads sequences with 0 and labels with -1; used as collate_fn of DataLoader.

    Loss functions will ignore -1 labels.
    If labels are 1D, pads to the maximum sequence length.
    If labels are 2D, pads all to (maxlen,maxlen).

    Args:
      batch_observations: A list of observations composing a batch

    Return:
      A tuple of:
          input batch, padded
          label batch, padded
          lengths-of-inputs batch, padded
          Observation batch (not padded)
    """
    seqs = [x[0].sentence for x in batch_observations]
    lengths = torch.tensor([len(x) for x in seqs], device="cpu")
    label_shape = batch_observations[0][1].shape
    maxlen = int(max(lengths))
    label_maxshape = [maxlen for x in label_shape]
    labels = [-torch.ones(*label_maxshape, device="cpu") for x in seqs]
    for index, x in enumerate(batch_observations):
        length = x[1].shape[0]
        if len(label_shape) == 1:
            labels[index][:length] = x[1]
        elif len(label_shape) == 2:
            labels[index][:length, :length] = x[1]
        else:
            raise ValueError("Labels must be either 1D or 2D right now; got either 0D or >3D")
    labels = torch.stack(labels)
    return seqs, labels, lengths, batch_observations


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def print_tikz(reporting_root, prediction_edges, gold_edges, edge_to_relation, words, split_name):
    """ Turns edge sets on word (nodes) into tikz dependency LaTeX. """
    words = list(words)
    for i, word in enumerate(words):
        word = word.replace("$", "\$").replace("&", "+").replace("%", "\%")
        if has_numbers(word):
            word = f"${word}$"
        words[i] = word

    with open(os.path.join(reporting_root, "visualize.tikz"), "a") as fout:
        string = "\\begin{figure}"
        string += "\\resizebox{\\textwidth}{!}{" + "\n"
        string += """\\begin{dependency}[edge unit distance=5ex]
\\begin{deptext}[column sep=2cm]
"""
        string += "\\& ".join([x for x in words]) + " \\\\" + "\n"
        string += "\\end{deptext}" + "\n"
        for i_index, j_index in gold_edges:
            string += "\\depedge{{{}}}{{{}}}{{{}}}\n".format(
                i_index + 1, j_index + 1, edge_to_relation.get((i_index, j_index), ".")
            )
        for i_index, j_index in prediction_edges:
            string += "\\depedge[edge style={{red!60!}}, edge below]{{{}}}{{{}}}{{{}}}\n".format(
                i_index + 1, j_index + 1, edge_to_relation.get((i_index, j_index), "wrong")
            )
        string += "\\end{dependency}\n"
        string += "}\n"
        string += "\\end{figure}"
        string += "\\clearpage"
        fout.write("\n\n")
        fout.write(string)


def convert_to_prediction(
    tokenizer,
    word_piece_pmi,
    raw_sen,
    pmi_clamp_zero,
    word_piece_agg_space,
    word_piece_agg_type,
    no_prefix_space=False,
):
    if pmi_clamp_zero:
        word_piece_pmi.clamp_(min=0)

    raw_words = raw_sen.split()
    if word_piece_pmi.size(0) == len(raw_words):
        word_pmi = word_piece_pmi
    else:
        if word_piece_agg_space == "exp":
            word_piece_pmi = word_piece_pmi.exp()
        temp_word_pmi = torch.zeros(word_piece_pmi.size(0), len(raw_words))
        word_piece_pt = 0
        word_to_num_pieces = []

        for word_pt, word in enumerate(raw_sen.split()):
            if "roberta" in tokenizer.__class__.__name__.lower():
                tokens = tokenizer.tokenize(word, add_prefix_space=not (no_prefix_space and word_pt == 0))
            else:
                tokens = tokenizer.tokenize(word)
            word_to_num_pieces.append(len(tokens))
            if len(tokens) > 1:
                if word_piece_agg_type == "avg":
                    temp_word_pmi[:, word_pt] = word_piece_pmi[:, word_piece_pt : word_piece_pt + len(tokens)].mean(
                        dim=1
                    )
                elif word_piece_agg_type == "max":
                    temp_word_pmi[:, word_pt] = word_piece_pmi[:, word_piece_pt : word_piece_pt + len(tokens)].max(
                        dim=1
                    )[0]
                elif word_piece_agg_type == "min":
                    temp_word_pmi[:, word_pt] = word_piece_pmi[:, word_piece_pt : word_piece_pt + len(tokens)].min(
                        dim=1
                    )[0]
            else:
                temp_word_pmi[:, word_pt] = word_piece_pmi[:, word_piece_pt]
            word_piece_pt += len(tokens)

        word_pmi = torch.zeros(len(raw_words), len(raw_words))
        word_piece_pt = 0
        for word_pt, num_pieces in enumerate(word_to_num_pieces):
            if num_pieces > 1:
                word_pmi[word_pt] = temp_word_pmi[word_piece_pt : word_piece_pt + num_pieces].mean(dim=0)
            else:
                word_pmi[word_pt] = temp_word_pmi[word_piece_pt]
            word_piece_pt += num_pieces

        if word_piece_agg_space == "exp":
            word_piece_pmi = word_piece_pmi.log()
            word_pmi = word_pmi.log()

    assert word_pmi.size(0) == len(raw_words), "Doesn't type check after alignment"

    prediction = -word_pmi  # convert to distance-like metric

    prediction = prediction.numpy().astype("double")

    for i in range(len(raw_words)):
        prediction[i][i] = float("-inf")
    for i in range(len(raw_words) - 1):
        prediction[i][i + 1] -= 1e-4  # default to right branching

    return prediction


def convert_word_piece_pmi_to_predictions(args, word_piece_pmis):
    with open(os.path.join(args.raw_data_dir, f"raw.{args.split}.txt"), "r") as f:
        raw_sens = f.readlines()
        raw_sens = [s.strip() for s in raw_sens]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    predictions = []
    for word_piece_pmi, raw_sen in tqdm(zip(word_piece_pmis, raw_sens), total=len(raw_sens), desc="aligning word pmi"):
        prediction = convert_to_prediction(
            tokenizer,
            word_piece_pmi,
            raw_sen,
            args.pmi_clamp_zero,
            args.word_piece_agg_space,
            args.word_piece_agg_type,
            args.no_prefix_space,
        )
        prediction = np.expand_dims(prediction, axis=0)  # stupid hack for the pipeline
        predictions.append(prediction)

    return predictions


Observations = namedtuple(
    "Observations",
    [
        "index",
        "sentence",
        "lemma_sentence",
        "upos_sentence",
        "xpos_sentence",
        "morph",
        "head_indices",
        "governance_relations",
        "secondary_relations",
        "extra_info",
        "embeddings",
    ],
)

if __name__ == "__main__":

    args = argp.parse_args()

    # Preparing output path
    out_path = os.path.join(args.output_dir)
    os.makedirs(out_path, exist_ok=True)

    if os.path.isfile(os.path.join(out_path, f"visualize.tikz")):
        os.remove(os.path.join(out_path, f"visualize.tikz"))
    args.output_dir = out_path

    dataset_cache_path = os.path.join(args.raw_data_dir, f"{args.split}.observations.dataset")
    if os.path.isfile(dataset_cache_path):
        dataset = torch.load(dataset_cache_path)
    else:
        observations_path = os.path.join(args.raw_data_dir, f"ptb3-wsj-{args.split}.conllx")
        observations = load_conll_dataset(observations_path)
        task = ParseDistanceTask()
        dataset = ObservationIterator(observations, task)
        torch.save(dataset, dataset_cache_path)

    # Loading test set data
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=custom_pad, shuffle=False)
    word_piece_pmi_path = args.word_piece_pmi_path
    word_piece_pmis = torch.load(word_piece_pmi_path)
    predictions = convert_word_piece_pmi_to_predictions(args, word_piece_pmis)
    torch.save(predictions, os.path.join(args.output_dir, "converted_predictions.pt"))

    # Heavy-lifting for the evaluation is done in this function
    (
        uuas,
        uuas_per_sen,
        uuas_adjacent,
        uuas_rand,
        per_relation_stats,
        per_relation_stats_adjacent,
    ) = report_uuas_and_tikz(args, predictions, dataloader, args.split)

    print("UUAS per relation")
    per_relation_analysis = sorted(per_relation_stats.items(), key=lambda x: x[1][0], reverse=True)
    per_relation_analysis_adjacent = sorted(per_relation_stats_adjacent.items(), key=lambda x: x[1][0], reverse=True)
    data = []
    for i in range(len(per_relation_analysis)):
        acc_mi = per_relation_analysis[i][1][1] / per_relation_analysis[i][1][0]
        acc_adjacent = per_relation_analysis_adjacent[i][1][1] / per_relation_analysis_adjacent[i][1][0]
        try:
            odds_ratio = math.log(acc_mi / (1 - acc_mi)) - math.log(acc_adjacent / (1 - acc_adjacent))
            se = math.sqrt(
                1 / per_relation_analysis[i][1][1]
                + 1 / per_relation_analysis_adjacent[i][1][1]
                + 1 / (per_relation_analysis[i][1][0] - per_relation_analysis[i][1][1])
                + 1 / (per_relation_analysis_adjacent[i][1][0] - per_relation_analysis_adjacent[i][1][1])
            )
        except:
            odds_ratio = 0
            se = 0

        data.append(
            (
                per_relation_analysis[i][0],
                per_relation_analysis[i][1][0],
                per_relation_analysis[i][1][1] / per_relation_analysis[i][1][0],
                per_relation_analysis_adjacent[i][1][1] / per_relation_analysis_adjacent[i][1][0],
                odds_ratio,
                1.96 * se,
            )
        )
    data = sorted(data, key=lambda x: x[2] - x[3], reverse=True)
    print("Relation, Count, Our, Adjacent, Odds Ratio, CI")
    for relation, count, num1, num2, log_odds_ratio, ci in data:
        print(
            f"{relation:10},",
            f"{count:4},",
            f"{num1:6.2%},",
            f"{num2:6.2%},",
            f"{log_odds_ratio:6.2f},",
            f"{ci:6.2f}",
        )

    # bootstrap confidence interval
    all_bootstraps = []
    for i in range(int(1e4)):
        bootstrap_uuas = np.mean(np.random.choice(uuas_per_sen, size=(len(uuas_per_sen), ), replace=True))
        all_bootstraps.append(bootstrap_uuas)

    # Printing final report
    print(f"Bootsrap mean {np.mean(all_bootstraps)}, std +- {np.std(all_bootstraps)}")
    print(f"UUAS: {uuas}")
    print(f"Right Branching UUAS: {uuas_adjacent}")
    print(f"Random Tree UUAS: {uuas_rand}")

    with open(os.path.join(args.output_dir, "uuas.txt"), "w") as f:
        f.write(f"Right Branching UUAS: {uuas_adjacent:.6f}" + "\n")
        f.write(f"UUAS: {uuas:.6f}" + "\n")
    torch.save(args, os.path.join(out_path, "args.pt"))
