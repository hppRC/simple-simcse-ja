import random
import unicodedata
from pathlib import Path

from tap import Tap

from src import utils


class Args(Tap):
    jsnli_dir: Path = "./datasets/nli/jsnli"
    nu_snli_dir: Path = "./datasets/nli/nu-snli"
    save_dir: Path = "./datasets/nli/jsnli+nu-snli"
    seed: int = 42


def main(args: Args):
    utils.set_seed(args.seed)

    jsnli = utils.load_jsonl(args.jsnli_dir / "train.jsonl").to_dict("records")
    nu_snli = utils.load_jsonl(args.nu_snli_dir / "train.jsonl").to_dict("records")
    combined = jsnli + nu_snli

    random.shuffle(combined)
    combined = [{**example, "id": idx} for idx, example in enumerate(combined)]
    combined = [{"id": example["id"], **example} for example in combined]
    utils.save_jsonl(combined, args.save_dir / "train.jsonl")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
