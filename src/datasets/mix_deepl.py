import random
from pathlib import Path

from classopt import classopt

from src import utils


@classopt(default_long=True)
class Args:
    dataset_dir: Path = "./datasets"
    save_dir: Path = "./datasets/nli/jsnli+deepl_mnli"
    seed: int = 42


def main(args: Args):
    utils.set_seed(args.seed)

    jsnli_all = utils.load_jsonl(args.dataset_dir / "nli/jsnli/train.jsonl").to_dict("records")
    # deepl_all = utils.load_jsonl(
    #     args.dataset_dir / "nli-translated/deepl/snli/train.jsonl"
    # ).to_dict("records")
    deepl_all = utils.load_jsonl(
        args.dataset_dir / "nli-translated/deepl/mnli/train.jsonl"
    ).to_dict("records")
    # deepl_all = utils.load_jsonl(
    #     args.dataset_dir / "nli-translated/deepl/snli+mnli/train.jsonl"
    # ).to_dict("records")
    data = jsnli_all + deepl_all
    random.shuffle(data)

    utils.save_jsonl(data, args.save_dir / "train.jsonl")


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
