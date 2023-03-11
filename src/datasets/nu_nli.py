import random
import unicodedata
from pathlib import Path

from tap import Tap

from src import utils


class Args(Tap):
    data_dir: Path = "./data/nu-nli"
    save_dir: Path = "./datasets/nli"
    seed: int = 42


def preprocess(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace(" ", "").strip())
    return text


def main(args: Args):
    utils.set_seed(args.seed)

    train_all = []

    for name in ["snli", "mnli"]:
        data_dir = args.data_dir / name

        for split in ["train", "val", "test"]:
            df = utils.load_jsonl(data_dir / f"{split}.jsonl")
            df = df.dropna(subset=["premise", "hypothesis"])

            data = []
            for _, group in df.groupby("group_id"):
                sent0 = group["premise"].unique().tolist()
                sent1 = group[group["label"] == "entailment"]["hypothesis"].unique().tolist()
                hard_neg = group[group["label"] == "contradiction"]["hypothesis"].unique().tolist()

                if len(sent0) == 0 or len(sent1) == 0:
                    continue

                data.append({"sent0": sent0, "sent1": sent1, "hard_neg": hard_neg})

            if split == "train":
                random.shuffle(data)
                train_all += data

            data = [{"id": idx, **example} for idx, example in enumerate(data)]

            utils.save_jsonl(data, args.save_dir / f"nu-{name}/{split}.jsonl")

    random.shuffle(train_all)
    train_all = [{"id": idx, **example} for idx, example in enumerate(train_all)]
    utils.save_jsonl(train_all, args.save_dir / f"nu-snli+mnli/train.jsonl")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
