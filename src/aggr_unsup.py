import json
from pathlib import Path

import pandas as pd
from classopt import classopt


@classopt(default_long=True)
class Args:
    input_dir: Path = "./outputs/prev/2022-11-29"
    output_dir: Path = "./outputs/results/unsup-simcse"


def main(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = []
    for sts_metrics_path in args.input_dir.glob("**/sts-metrics.json"):
        dir = sts_metrics_path.parent
        with (dir / "config.json").open() as f:
            config = json.load(f)
            model_name = config["model_name"]

        with sts_metrics_path.open() as f:
            sts_metrics = json.load(f)

        data.append(
            {
                "model_name": model_name,
                "dataset_name": "wiki40b",
                **sts_metrics,
            }
        )

    df = pd.DataFrame(data).sort_values(by="avg", ascending=False)
    df.to_csv(args.output_dir / "all.csv", index=False)


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
