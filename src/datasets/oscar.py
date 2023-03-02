import os
import random
from pathlib import Path

from classopt import classopt
from more_itertools import flatten

from datasets import DatasetDict, load_dataset
from datasets.formatting.formatting import LazyBatch
from src.datasets.common import preprocess_text


@classopt(default_long=True)
class Args:
    output_dir: Path = "./datasets/text/oscar"
    num_train_examples: int = 1_000_000
    seed: int = 42


def process(text: str) -> list[str]:
    sentences: list[str] = [s.strip() for s in text.splitlines()]
    sentences: list[str] = list(flatten(preprocess_text(s) for s in sentences))
    return sentences


def batch_process(batch: LazyBatch):
    sentences = [process(text) for text in batch["text"]]
    return {
        "sentences": sentences,
    }


def main(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    datasets: DatasetDict = load_dataset(
        "oscar",
        name="unshuffled_deduplicated_ja",
    )

    datasets = datasets.map(
        batch_process,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count(),
        remove_columns=["text"],
    )
    print(sum(len(e["sentences"]) for e in datasets["train"]))
    datasets.save_to_disk(args.output_dir)
    print("done loading dataset")

    sentences = []
    for example in datasets["train"]:
        sentences += example["sentences"]

    (args.output_dir / "all.txt").write_text("\n".join(sentences))
    train_examples = random.sample(sentences, args.num_train_examples)
    (args.output_dir / "train.txt").write_text("\n".join(train_examples))


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
