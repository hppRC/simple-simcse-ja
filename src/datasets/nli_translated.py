import math
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

from classopt import classopt

from src import utils


@classopt(default_long=True)
class Args:
    input_dir: Path = "./data/nli-translated"
    save_dir: Path = "./datasets/nli-translated"
    method: str = "nllb"
    seed: int = 42


def normalize(text: str) -> str:
    text = text.replace(" ", "").replace("。", "").replace("、", "").strip()
    text = text.replace(".", "").replace(",", "").strip()
    text = unicodedata.normalize("NFKC", text)
    return text


def preprocess(text: str) -> str:
    text = text.replace(" ", "").strip()
    text = unicodedata.normalize("NFKC", text)
    return text


def snli(input_dir: Path, save_dir: Path):
    for src_name, dst_name in [
        ("train", "train"),
        ("dev", "val"),
        ("test", "test"),
    ]:
        df = utils.load_jsonl(input_dir / f"snli/{src_name}.jsonl")
        df = df.dropna(subset=["sentence1", "sentence2", "gold_label"])

        examples = defaultdict(lambda: defaultdict(set))

        for premise, hypothesis, label in df[["sentence1", "sentence2", "gold_label"]].values:
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

        utils.save_jsonl(data, save_dir / f"snli/{dst_name}.jsonl")
    return utils.load_jsonl(save_dir / f"snli/train.jsonl").to_dict("records")


def mnli(input_dir: Path, save_dir: Path):
    for src_name, dst_name in [
        ("train", "train"),
        ("dev_matched", "val"),
        ("dev_mismatched", "test"),
    ]:
        df = utils.load_jsonl(input_dir / f"mnli/{src_name}.jsonl")
        df = df.dropna(subset=["sentence1", "sentence2", "gold_label"])

        examples = defaultdict(lambda: defaultdict(set))

        for premise, hypothesis, label in df[["sentence1", "sentence2", "gold_label"]].values:
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

        utils.save_jsonl(data, save_dir / f"mnli/{dst_name}.jsonl")
    return utils.load_jsonl(save_dir / f"mnli/train.jsonl").to_dict("records")


def main(args: Args):
    utils.set_seed(args.seed)

    train_all = []

    train_all += snli(args.input_dir / args.method, args.save_dir / args.method)
    train_all += mnli(args.input_dir / args.method, args.save_dir / args.method)

    random.shuffle(train_all)
    train_all = [{**example, "id": idx} for idx, example in enumerate(train_all)]
    utils.save_jsonl(train_all, args.save_dir / args.method / "snli+mnli" / "train.jsonl")


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
