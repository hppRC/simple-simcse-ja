import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import pandas as pd
from more_itertools import flatten
from src import utils
from tap import Tap

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
                f"| {d['dataset_name']} "
                f"| {d['best-dev']:.2f} | {d['jsick']:.2f} "
                f"| {d['jsts-train']:.2f} | {d['jsts-val']:.2f} "
                f"| {d['avg']:.2f} |"
            )


def model_rank(args: Args):
    print("model_rank")
    overall = {}

    for method in ["sup-simcse", "unsup-simcse"]:
        print(method)
        path = args.input_dir / method / "best.csv"
        df = pd.read_csv(path)
        data = defaultdict(dict)

        for dataset_name, group_df in df.groupby("dataset_name"):
            group_df = group_df.sort_values("avg", ascending=False)
            for rank, model_name in enumerate(group_df["model_name"].tolist()):
                data[dataset_name][model_name] = rank + 1

        data: pd.Series = pd.DataFrame(data).mean(axis=1)
        overall[method] = data

    overall = pd.DataFrame(overall).sort_values(by="sup-simcse", ascending=True)
    print(overall)


def dataset_rank(args: Args):
    print("dataset_rank")

    for method in ["sup-simcse", "unsup-simcse"]:
        path = args.input_dir / method / "best.csv"
        df = pd.read_csv(path)
        data = defaultdict(dict)

        for model_name, group_df in df.groupby("model_name"):
            group_df = group_df.sort_values("avg", ascending=False)
            for rank, dataset_name in enumerate(group_df["dataset_name"].tolist()):
                data[model_name][dataset_name] = rank + 1

        data: pd.Series = pd.DataFrame(data).mean(axis=1).sort_values(ascending=True)
        print(data)


def batch_rank(args: Args):
    print("batch_rank")

    for method in ["sup-simcse", "unsup-simcse"]:
        path = args.input_dir / method / "all.csv"
        df = pd.read_csv(path)
        data_df = {}

        data = defaultdict(lambda: defaultdict(dict))
        for key, group_df in df.groupby(["model_name", "dataset_name"]):
            for lr, lr_group in group_df.groupby("lr"):
                lr_group = lr_group.sort_values("avg", ascending=False)
                for rank, batch_size in enumerate(lr_group["batch_size"].tolist()):
                    data[lr][key][batch_size] = rank + 1
        for lr in data.keys():
            data_df[lr] = (
                pd.DataFrame(data[lr]).mean(axis=1).sort_values(ascending=True)
            )
        data_df["avg"] = pd.DataFrame(data_df).mean(axis=1).sort_values(ascending=True)
        df = pd.DataFrame(data_df)
        print(df)


def counts(args: Args):
    print("counts")

    for method in ["sup-simcse", "unsup-simcse"]:
        path = args.input_dir / method / "all.csv"
        df = pd.read_csv(path)
        print(df["count"].sum())


if __name__ == "__main__":
    args = Args().parse_args()
    by_models(args)
    by_datasets(args)
    model_rank(args)
    dataset_rank(args)
    batch_rank(args)
    counts(args)
