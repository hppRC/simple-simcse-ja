import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from classopt import classopt

from src import utils


@classopt(default_long=True)
class Args:
    data_dir: Path = "./data"
    save_dir: Path = "./datasets/nli"
    seed: int = 42


def normalize(text: str) -> str:
    text = text.replace(" ", "").replace("。", "").replace("、", "").strip()
    text = unicodedata.normalize("NFKC", text)
    return text


def preprocess(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace(" ", "").strip())
    return text


def janli_test(data_dir: Path, save_dir: Path):
    df = pd.read_table(data_dir / "janli" / "janli.tsv")
    df = df[df["split"] == "test"]
    examples = defaultdict(lambda: defaultdict(set))

    for premise, hypothesis, label in df[
        ["sentence_A_Ja", "sentence_B_Ja", "entailment_label_Ja"]
    ].values:
        identifier = normalize(preprocess(premise))
        examples[identifier]["premise"].add(preprocess(premise))
        examples[identifier][label].add(preprocess(hypothesis))

    data = []
    for example_dict in examples.values():
        sent0: list[str] = list(example_dict["premise"])
        sent1: list[str] = list(example_dict["entailment"])
        hard_neg: list[str] = list(example_dict["non-entailment"])
        if len(sent0) == 0 or len(sent1) == 0:
            continue
        data.append({"sent0": sent0, "sent1": sent1, "hard_neg": hard_neg})
    random.shuffle(data)
    data = [{"id": idx, **example} for idx, example in enumerate(data)]

    utils.save_jsonl(data, save_dir / "janli/test.jsonl")
    return data


def janli_train(data_dir: Path, save_dir: Path):
    df = pd.read_table(data_dir / "janli" / "janli.tsv")
    df = df[df["split"] == "train"]
    examples = defaultdict(lambda: defaultdict(set))

    for premise, hypothesis, label in df[
        ["sentence_A_Ja", "sentence_B_Ja", "entailment_label_Ja"]
    ].values:
        identifier = normalize(preprocess(premise))
        examples[identifier]["premise"].add(preprocess(premise))
        examples[identifier][label].add(preprocess(hypothesis))

    data = []
    for example_dict in examples.values():
        sent0: list[str] = list(example_dict["premise"])
        sent1: list[str] = list(example_dict["entailment"])
        hard_neg: list[str] = list(example_dict["non-entailment"])
        if len(sent0) == 0 or len(sent1) == 0:
            continue
        data.append({"sent0": sent0, "sent1": sent1, "hard_neg": hard_neg})
    random.shuffle(data)
    data = [{"id": idx, **example} for idx, example in enumerate(data)]

    utils.save_jsonl(data, save_dir / "janli/train.jsonl")
    return data


def jnli_train(data_dir: Path, save_dir: Path):
    df = utils.load_jsonl(data_dir / "jnli/train-v1.1.json")
    examples = defaultdict(lambda: defaultdict(set))

    for premise, hypothesis, label in df[["sentence1", "sentence2", "label"]].values:
        identifier = normalize(preprocess(premise))
        examples[identifier]["premise"].add(preprocess(premise))
        examples[identifier][label].add(preprocess(hypothesis))

    data = []
    for example_dict in examples.values():
        sent0: list[str] = list(example_dict["premise"])
        sent1: list[str] = list(example_dict["entailment"])
        hard_neg: list[str] = list(example_dict["contradiction"])
        if len(sent0) == 0 or len(sent1) == 0:
            continue
        data.append({"sent0": sent0, "sent1": sent1, "hard_neg": hard_neg})
    random.shuffle(data)
    data = [{"id": idx, **example} for idx, example in enumerate(data)]

    utils.save_jsonl(data, save_dir / "jnli/train.jsonl")
    return data


def jnli_val(data_dir: Path, save_dir: Path):
    df = utils.load_jsonl(data_dir / "jnli/valid-v1.1.json")
    examples = defaultdict(lambda: defaultdict(set))

    for premise, hypothesis, label in df[["sentence1", "sentence2", "label"]].values:
        identifier = normalize(preprocess(premise))
        examples[identifier]["premise"].add(preprocess(premise))
        examples[identifier][label].add(preprocess(hypothesis))

    data = []
    for example_dict in examples.values():
        sent0: list[str] = list(example_dict["premise"])
        sent1: list[str] = list(example_dict["entailment"])
        hard_neg: list[str] = list(example_dict["contradiction"])
        if len(sent0) == 0 or len(sent1) == 0:
            continue
        data.append({"sent0": sent0, "sent1": sent1, "hard_neg": hard_neg})
    random.shuffle(data)
    data = [{"id": idx, **example} for idx, example in enumerate(data)]

    utils.save_jsonl(data, save_dir / "jnli/val.jsonl")
    return data


def jsnli_train(data_dir: Path, save_dir: Path):
    df = pd.read_table(
        data_dir / "jsnli/train_w_filtering.tsv",
        header=None,
        names=["label", "premise", "hypothesis"],
    )
    examples = defaultdict(lambda: defaultdict(set))

    for premise, hypothesis, label in df[["premise", "hypothesis", "label"]].values:
        premise = preprocess(premise)
        hypothesis = preprocess(hypothesis)
        identifier = normalize(premise)

        examples[identifier]["premise"].add(premise)
        examples[identifier][label].add(hypothesis)

    data = []
    for example_dict in examples.values():
        sent0: list[str] = list(example_dict["premise"])
        sent1: list[str] = list(example_dict["entailment"])
        hard_neg: list[str] = list(example_dict["contradiction"])
        if len(sent0) == 0 or len(sent1) == 0:
            continue
        data.append({"sent0": sent0, "sent1": sent1, "hard_neg": hard_neg})
    random.shuffle(data)
    data = [{"id": idx, **example} for idx, example in enumerate(data)]

    utils.save_jsonl(data, save_dir / "jsnli/train.jsonl")
    return data


def jsnli_val(data_dir: Path, save_dir: Path):
    df = pd.read_table(
        data_dir / "jsnli/dev.tsv",
        header=None,
        names=["label", "premise", "hypothesis"],
    )
    examples = defaultdict(lambda: defaultdict(set))

    for premise, hypothesis, label in df[["premise", "hypothesis", "label"]].values:
        premise = preprocess(premise)
        hypothesis = preprocess(hypothesis)

        identifier = normalize(premise)
        examples[identifier]["premise"].add(premise)
        examples[identifier][label].add(hypothesis)

    data = []
    for identifier, example_dict in examples.items():
        sent0: list[str] = list(example_dict["premise"])
        sent1: list[str] = list(example_dict["entailment"])
        hard_neg: list[str] = list(example_dict["contradiction"])
        if len(sent0) == 0 or len(sent1) == 0:
            continue
        data.append({"sent0": sent0, "sent1": sent1, "hard_neg": hard_neg})
    random.shuffle(data)
    data = [{"id": idx, **example} for idx, example in enumerate(data)]

    utils.save_jsonl(data, save_dir / "jsnli/val.jsonl")
    return data


def main(args: Args):
    utils.set_seed(args.seed)

    train_all = []

    train_all += janli_train(args.data_dir, args.save_dir)
    janli_test(args.data_dir, args.save_dir)

    train_all += jnli_train(args.data_dir, args.save_dir)
    jnli_val(args.data_dir, args.save_dir)

    train_all += jsnli_train(args.data_dir, args.save_dir)
    jsnli_val(args.data_dir, args.save_dir)




if __name__ == "__main__":
    args = Args.from_args()
    main(args)
