import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import pandas as pd
from more_itertools import flatten
from src import utils
from tap import Tap

LEARNING_RATES = [1e-5, 3e-5, 5e-5]
BATCH_SIZES = [64, 128, 256, 512]

SUP_DATASETS = ["jsnli", "janli", "nu-snli", "nu-mnli", "nu-snli+mnli"]
UNSUP_DATASETS = ["wikipedia", "wiki40b", "bccwj", "cc100"]

DATASETS = SUP_DATASETS + UNSUP_DATASETS

BASE_MODELS = [
    "cl-tohoku/bert-base-japanese-v2",
    "cl-tohoku/bert-base-japanese-char-v2",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "ku-nlp/roberta-base-japanese-char-wwm",
    "studio-ousia/luke-japanese-base-lite",
    "ku-nlp/deberta-v2-base-japanese",
    "nlp-waseda/roberta-base-japanese",
    "megagonlabs/roberta-long-japanese",
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "studio-ousia/mluke-base-lite",
    "cl-tohoku/bert-base-japanese-v3",
    "cl-tohoku/bert-base-japanese-char-v3",
]

LARGE_MODELS = [
    "cl-tohoku/bert-large-japanese",
    "ku-nlp/roberta-large-japanese-char-wwm",
    "studio-ousia/luke-japanese-large-lite",
    "nlp-waseda/roberta-large-japanese",
    "ku-nlp/deberta-v2-large-japanese",
    "xlm-roberta-large",
    "studio-ousia/mluke-large-lite",
    "cl-tohoku/bert-large-japanese-v2",
    "cl-tohoku/bert-large-japanese-char-v2",
]

LORA_MODELS = [
    "google/mt5-large",
    "google/mt5-xl",
    "google/mt5-xxl",
]

OTHER_MODELS = []


MODELS = BASE_MODELS + LARGE_MODELS
# MODELS = BASE_MODELS + LARGE_MODELS + LORA_MODELS + OTHER_MODELS


class Args(Tap):
    input_dir: Path = "./outputs"
    output_dir: Path = "./outputs/results"
    num_experiments: int = 5


def extract(sts_metrics_path: Path) -> tuple[str, dict]:
    try:
        dir = sts_metrics_path.parent
        with (dir / "config.json").open() as f:
            config = json.load(f)

        method = config.get("METHOD")

        if method == "lora-sup-simcse":
            return method, None

        model_name = config.get("model_name")

        if model_name in LORA_MODELS:
            return method, None

        dataset_name = config.get("dataset_name")
        batch_size = config.get("batch_size")
        lr = str(config.get("lr"))

        max_seq_len = config.get("max_seq_len")

        if max_seq_len != 64:
            return method, None

        if not batch_size in BATCH_SIZES:
            return method, None

        if not dataset_name in DATASETS:
            return method, None

        with sts_metrics_path.open() as f:
            sts_metrics = json.load(f)
        with (dir / "dev-metrics.json").open() as f:
            dev_metrics = json.load(f)

        data = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "lr": lr,
            **dev_metrics,
            **sts_metrics,
        }

        return method, data
    except Exception as e:
        print(e)
        return method, None


def complete_experiments(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "sup-simcse":
        DATASETS = SUP_DATASETS
    elif method == "unsup-simcse":
        DATASETS = UNSUP_DATASETS

    completions = {
        (str(lr), batch_size, model_name, dataset_name): {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "lr": str(lr),
            "count": 0,
        }
        for lr, batch_size, model_name, dataset_name in product(
            LEARNING_RATES, BATCH_SIZES, MODELS, DATASETS
        )
    }

    for data in df.to_dict(orient="records"):
        model_name = data.get("model_name")
        dataset_name = data.get("dataset_name")
        batch_size = data.get("batch_size")
        lr = str(data.get("lr"))
        completions[(lr, batch_size, model_name, dataset_name)] = data

    return pd.DataFrame(completions.values())


def main(args: Args):
    data = defaultdict(list)

    dirs = [dir for dir in args.input_dir.glob("*") if not str(dir).startswith("outputs/prev")]
    paths = list(flatten(dir.glob("**/sts-metrics.json") for dir in dirs))

    with ThreadPoolExecutor(max_workers=256) as executor:
        for method, row in executor.map(extract, paths):
            if row is None:
                continue
            data[method].append(row)

    setting_columns = ["model_name", "dataset_name"]
    hparams_columns = ["batch_size", "lr"]
    config_columns = setting_columns + hparams_columns

    for method, df in data.items():
        output_dir = args.output_dir / method

        df = pd.DataFrame(df).sort_values(by=config_columns, ascending=False)

        df = (
            df.groupby(config_columns, as_index=False)
            .apply(lambda group: group.head(args.num_experiments).reset_index(drop=True))
            .reset_index(drop=True)
        )
        print(method, len(df))
        df = df.groupby(config_columns, as_index=False).agg(
            **{col: (col, "mean") for col in df.select_dtypes(include="number").columns},
            count=("dataset_name", "size"),
        )
        print(method, len(df))
        print(
            len(df[df["lr"].str.startswith("1e-05")]),
            len(df[df["lr"].str.startswith("3e-05")]),
            len(df[df["lr"].str.startswith("5e-05")]),
        )

        completed_df = complete_experiments(df, method=method)
        completed_df = completed_df.sort_values(by=config_columns, ascending=False)
        utils.save_csv(completed_df, output_dir / "all.csv")

        rest_df = completed_df[completed_df["count"] < args.num_experiments]
        utils.save_csv(rest_df, output_dir / "rest.csv")
        print(sum(args.num_experiments - c for c in rest_df["count"].tolist()))

        best_df = (
            df.groupby(setting_columns, as_index=False)
            .apply(lambda x: x.nlargest(1, "best-dev").reset_index(drop=True))
            .reset_index(drop=True)
        ).sort_values("avg", ascending=False)

        utils.save_csv(best_df, output_dir / "best.csv")

        for dataset_name in df["dataset_name"].unique():
            dataset_output_dir = output_dir / "datasets" / dataset_name
            utils.save_csv(df, dataset_output_dir / "all.csv", "dataset_name", dataset_name)
            utils.save_csv(best_df, dataset_output_dir / "best.csv", "dataset_name", dataset_name)

        for model_name in df["model_name"].unique():
            model_output_dir = output_dir / "models" / model_name
            utils.save_csv(df, model_output_dir / "all.csv", "model_name", model_name)
            utils.save_csv(best_df, model_output_dir / "best.csv", "model_name", model_name)


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
