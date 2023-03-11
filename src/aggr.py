import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from more_itertools import flatten
from tap import Tap


class Args(Tap):
    input_dir: Path = "./outputs"
    output_dir: Path = "./outputs/results"


def main(args: Args):
    data = defaultdict(list)

    dirs = [dir for dir in args.input_dir.glob("*") if not str(dir).startswith("outputs/prev")]
    paths = list(flatten(dir.glob("**/sts-metrics.json") for dir in dirs))

    for sts_metrics_path in paths:
        try:
            dir = sts_metrics_path.parent
            with (dir / "config.json").open() as f:
                config = json.load(f)

            method = config.get("METHOD")

            model_name = config.get("model_name")
            dataset_name = config.get("dataset_name")
            batch_size = config.get("batch_size")
            lr = config.get("lr")

            max_seq_len = config.get("max_seq_len")

            if max_seq_len != 64:
                continue

            if not batch_size in [64, 128, 256, 512]:
                continue

            # if not dataset_name in ["jsnli", "nu-snli"]:
            #     continue

            with sts_metrics_path.open() as f:
                sts_metrics = json.load(f)
            with (dir / "dev-metrics.json").open() as f:
                dev_metrics = json.load(f)

            data[method].append(
                {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "batch_size": batch_size,
                    "lr": lr,
                    **dev_metrics,
                    **sts_metrics,
                }
            )

        except Exception as e:
            print(e)
            continue

    setting_columns = ["model_name", "dataset_name"]
    hparams_columns = ["batch_size", "lr"]
    config_columns = setting_columns + hparams_columns

    for method, data in data.items():
        output_dir = args.output_dir / method
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(data)
        df = df.sort_values(by=config_columns, ascending=False)
        counts: list[int] = list(df.groupby(config_columns).size())

        df = df.groupby(config_columns, as_index=False).mean(numeric_only=True)
        df["count"] = counts
        df.to_csv(output_dir / "all.csv", index=False)

        best_df = (
            df.groupby(setting_columns, as_index=False)
            .apply(lambda x: x.nlargest(1, "best-dev").reset_index(drop=True))
            .reset_index(drop=True)
        )
        best_df = best_df.sort_values("avg", ascending=False)
        best_df.to_csv(output_dir / "best.csv", index=False)

        for dataset_name in df["dataset_name"].unique():
            (output_dir / dataset_name).mkdir(parents=True, exist_ok=True)
            df[df["dataset_name"] == dataset_name].to_csv(
                output_dir / dataset_name / "all.csv",
                index=False,
            )
            best_df[best_df["dataset_name"] == dataset_name].to_csv(
                output_dir / dataset_name / "best.csv",
                index=False,
            )


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
