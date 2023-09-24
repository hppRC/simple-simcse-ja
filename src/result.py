from collections import defaultdict
from pathlib import Path

import pandas as pd
from tap import Tap


class Args(Tap):
    input_dir: Path = "./outputs/results"


def by_models(args: Args):
    overall = {}

    for method in ["sup-simcse", "unsup-simcse"]:
        df = pd.read_csv(args.input_dir / method / "best.csv")
        data = defaultdict(dict)

        for dataset_name, group_df in df.groupby("dataset_name"):
            group_df = group_df.sort_values("avg", ascending=False)
            for rank, model_name in enumerate(group_df["model_name"].tolist()):
                reciprocal_rank = 1 / (rank + 1)
                data[dataset_name][model_name] = reciprocal_rank

        data: pd.Series = pd.DataFrame(data).mean(axis=1)
        overall[method] = data

    overall = pd.DataFrame(overall).sort_values(by="sup-simcse", ascending=False)
    print(overall)


def by_datasets(args: Args):
    for method in ["sup-simcse", "unsup-simcse"]:
        df = pd.read_csv(args.input_dir / method / "best.csv")
        data = defaultdict(dict)

        for model_name, group_df in df.groupby("model_name"):
            group_df = group_df.sort_values("avg", ascending=False)
            for rank, dataset_name in enumerate(group_df["dataset_name"].tolist()):
                reciprocal_rank = 1 / (rank + 1)
                data[model_name][dataset_name] = reciprocal_rank

        data: pd.Series = pd.DataFrame(data).mean(axis=1).sort_values(ascending=False)
        print(method, data)


def main(args: Args):
    by_models(args)
    by_datasets(args)


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
