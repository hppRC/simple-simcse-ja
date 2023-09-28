import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import pandas as pd
from more_itertools import flatten
from tap import Tap

from src import utils

SUP_DATASETS = ["jsnli", "janli", "nu-snli", "nu-mnli", "nu-snli+mnli"]
UNSUP_DATASETS = ["wikipedia", "wiki40b", "bccwj", "cc100"]

BASE_MODELS = [
    "cl-tohoku/bert-base-japanese-v3",
    "cl-tohoku/bert-base-japanese-v2",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "studio-ousia/luke-japanese-base-lite",
    "ku-nlp/deberta-v2-base-japanese",
    "nlp-waseda/roberta-base-japanese",
    "megagonlabs/roberta-long-japanese",
    "cl-tohoku/bert-base-japanese-char-v3",
    "cl-tohoku/bert-base-japanese-char-v2",
    "cl-tohoku/bert-base-japanese-char",
    "ku-nlp/roberta-base-japanese-char-wwm",
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "studio-ousia/mluke-base-lite",
]

LARGE_MODELS = [
    "cl-tohoku/bert-large-japanese-v2",
    "cl-tohoku/bert-large-japanese",
    "studio-ousia/luke-japanese-large-lite",
    "nlp-waseda/roberta-large-japanese",
    "ku-nlp/deberta-v2-large-japanese",
    "cl-tohoku/bert-large-japanese-char-v2",
    "ku-nlp/roberta-large-japanese-char-wwm",
    "xlm-roberta-large",
    "studio-ousia/mluke-large-lite",
]


class Args(Tap):
    input_dir: Path = "./results"


def by_models(args: Args):
    print("by models")
    for method in ["sup-simcse", "unsup-simcse"]:
        path = args.input_dir / method / "best.csv"
        df = pd.read_csv(path)

        if method == "sup-simcse":
            df = df[df["dataset_name"] == "jsnli"]
        elif method == "unsup-simcse":
            df = df[df["dataset_name"] == "wiki40b"]

        df = (
            df.groupby("model_name", as_index=False)
            .apply(lambda group: group.nlargest(1, "best-dev"))
            .reset_index(drop=True)
        )
        records = {d["model_name"]: d for d in df.to_dict(orient="records")}

        for models in [BASE_MODELS, LARGE_MODELS]:
            print(f"##")
            for model_name in models:
                d = records[model_name]
                print(
                    f"| [{model_name}](https://huggingface.co/{model_name}) "
                    f"| {d['best-dev']:.2f} | {d['jsick']:.2f} "
                    f"| {d['jsts-train']:.2f} | {d['jsts-val']:.2f} "
                    f"| {d['avg']:.2f} |"
                )


def by_datasets(args: Args):
    print("by datasets")
    for method in ["sup-simcse", "unsup-simcse"]:
        print(method)
        path = args.input_dir / method / "best.csv"
        df = pd.read_csv(path)
        df = df[df["model_name"] == "cl-tohoku/bert-large-japanese-v2"]

        df = (
            df.groupby("dataset_name", as_index=False)
            .apply(lambda group: group.nlargest(1, "best-dev"))
            .reset_index(drop=True)
        ).sort_values("best-dev", ascending=False)

        for d in df.to_dict("records"):
            print(
                f"| {d['dataset_name']}"
                f"| {d['best-dev']:.2f} | {d['jsick']:.2f} "
                f"| {d['jsts-train']:.2f} | {d['jsts-val']:.2f} "
                f"| {d['avg']:.2f} |"
            )


if __name__ == "__main__":
    args = Args().parse_args()
    by_models(args)
    by_datasets(args)