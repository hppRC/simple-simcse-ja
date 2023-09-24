import random
import subprocess
from pathlib import Path

import pandas as pd
from tap import Tap

JUMANPP_MODELS = {
    "ku-nlp/deberta-v2-base-japanese",
    "nlp-waseda/roberta-base-japanese",
    "megagonlabs/roberta-long-japanese",
    "nlp-waseda/roberta-large-japanese",
    "ku-nlp/deberta-v2-large-japanese",
}

LORA_MODELS = {
    "google/mt5-large",
    "google/mt5-xl",
    "google/mt5-xxl",
}


class Args(Tap):
    input_dir: Path = "./outputs/results"
    device: str = "cuda:0"
    dtype: str = "bf16"
    gradient_checkpointing: bool = False
    update_results: bool = False


def main(args: Args):
    while True:
        completed_counts = 0
        for method in ["sup-simcse", "unsup-simcse"]:
            df = pd.read_csv(args.input_dir / method / "rest.csv")

            if len(df) == 0:
                completed_counts += 1
                continue

            row = random.choice(df.to_dict("records"))

            model_name = row["model_name"]
            dataset_name = row["dataset_name"]
            batch_size = int(row["batch_size"])
            lr = float(row["lr"])

            match method:
                case "sup-simcse":
                    path = "src/train_sup.py"
                case "unsup-simcse":
                    path = "src/train_unsup.py"
                case _:
                    raise ValueError(f"Invalid method: {method}")

            cmd = f"poetry run python {path} --dataset_name {dataset_name} --model_name {model_name} --batch_size {batch_size} --lr {lr} --dtype {args.dtype} --device {args.device}".split()

            if model_name in JUMANPP_MODELS:
                cmd.append("--use_jumanpp")
            if model_name in LORA_MODELS:
                cmd.append("--use_lora")
            if args.gradient_checkpointing or batch_size >= 256:
                cmd.append("--gradient_checkpointing")

            print(" ".join(cmd))
            result = subprocess.run(cmd)
            print(result)

        if completed_counts == 2:
            break

        if args.update_results:
            result = subprocess.run("poetry run python src/aggr.py".split())


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
