import os
import random
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

from classopt import classopt
from more_itertools import flatten

from src.datasets.common import preprocess_text


@classopt(default_long=True)
class Args:
    # place your dataset directory in ./data
    # before running this script, you should download and unzip the dataset
    # FYI: https://zenn.dev/pinto0309/articles/c6f38abd082000
    input_dir: Path = "./data/BCCWJ/BCCWJ15DISC1NT/C-XML/VARIABLE"
    output_dir: Path = "./datasets/text/bccwj"
    num_train_examples: int = 1_000_000
    seed: int = 42


def process(path: Path) -> list[str]:
    xml = ET.parse(path)
    raw_text = ET.tostring(xml.getroot(), encoding="unicode", method="text")

    sentences: list[str] = [s.strip() for s in raw_text.splitlines()]
    sentences: list[str] = list(flatten(preprocess_text(s) for s in sentences))
    return sentences


def main(args: Args):
    args.output_dir.mkdir(exist_ok=True, parents=True)
    random.seed(args.seed)

    sentences = []
    with Pool(processes=os.cpu_count()) as pool:
        paths = args.input_dir.glob("**/*.xml")
        for ret in pool.imap_unordered(process, paths):
            sentences += ret

    (args.output_dir / "all.txt").write_text("\n".join(sentences))
    train_examples = random.sample(sentences, args.num_train_examples)
    (args.output_dir / "train.txt").write_text("\n".join(train_examples))


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
