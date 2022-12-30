from pathlib import Path

import pandas as pd
from classopt import classopt

from src import utils


@classopt(default_long=True)
class Args:
    data_dir: Path = "./data"
    save_dir: Path = "./datasets/sts"


def jsick_test(data_dir: Path, save_dir: Path):
    df = pd.read_table(data_dir / "jsick/jsick/jsick.tsv", sep="\t")
    df = df[df["data"] == "test"]
    df = pd.DataFrame(
        {
            "id": range(len(df)),
            "sent0": df["sentence_A_Ja"].values,
            "sent1": df["sentence_B_Ja"].values,
            "score": df["relatedness_score_Ja"].values,
        }
    )
    utils.save_jsonl(df, save_dir / "jsick/test.jsonl")


def jsick_train(data_dir: Path, save_dir: Path):
    df = pd.read_table(data_dir / "jsick/jsick/jsick.tsv", sep="\t")
    df = df[df["data"] == "train"]
    df = pd.DataFrame(
        {
            "id": range(len(df)),
            "sent0": df["sentence_A_Ja"].values,
            "sent1": df["sentence_B_Ja"].values,
            "score": df["relatedness_score_Ja"].values,
        }
    )
    utils.save_jsonl(df, save_dir / "jsick/train.jsonl")


def jsts_valid(data_dir: Path, save_dir: Path):
    df = pd.read_json(data_dir / "jsts/valid-v1.1.json", lines=True)
    df = pd.DataFrame(
        {
            "id": range(len(df)),
            "sent0": df["sentence1"].values,
            "sent1": df["sentence2"].values,
            "score": df["label"].values,
        }
    )
    utils.save_jsonl(df, save_dir / "jsts/val.jsonl")


def jsts_train(data_dir: Path, save_dir: Path):
    df = pd.read_json(data_dir / "jsts/train-v1.1.json", lines=True)
    df = pd.DataFrame(
        {
            "id": range(len(df)),
            "sent0": df["sentence1"].values,
            "sent1": df["sentence2"].values,
            "score": df["label"].values,
        }
    )
    utils.save_jsonl(df, save_dir / "jsts/train.jsonl")


def main(args: Args):
    jsick_test(args.data_dir, args.save_dir)
    jsick_train(args.data_dir, args.save_dir)
    jsts_valid(args.data_dir, args.save_dir)
    jsts_train(args.data_dir, args.save_dir)


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
