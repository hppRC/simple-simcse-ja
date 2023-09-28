import os
import random
from pathlib import Path

from classopt import classopt
from datasets import DatasetDict, load_dataset
from more_itertools import flatten
from src.datasets.common import preprocess_text


@classopt(default_long=True)
class Args:
    output_dir: Path = "./datasets/text/wiki40b"
    num_train_examples: int = 1_000_000
    seed: int = 42


# entries of Wiki-40B are saved as a list of articles
# each article is a raw text, which have some special tags
# for convinience, we preprocess the raw text into a list of sentences
def process(text: str):
    lines: list[str] = [l for l in text.splitlines() if l]

    assert lines[0] == "_START_ARTICLE_"

    title = lines[1]
    lines = lines[2:]

    sentences: list[str] = []
    line_id = 0

    while line_id < len(lines):
        if lines[line_id] == "_START_SECTION_":
            line_id += 2
        elif lines[line_id] == "_START_PARAGRAPH_":
            line_id += 1
        else:
            ss: list[str] = [s.strip() for s in lines[line_id].split("_NEWLINE_")]
            ss: list[str] = list(flatten(preprocess_text(s) for s in ss))

            sentences += ss
            line_id += 1

    return {
        "title": title,
        "sentences": sentences,
    }


def batch_process(batch):
    examples = [process(text) for text in batch["text"]]
    return {
        "title": [e["title"] for e in examples],
        "sentences": [e["sentences"] for e in examples],
    }


def main(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    datasets: DatasetDict = load_dataset("wiki40b", "ja", beam_runner="DirectRunner")

    datasets = datasets.map(
        batch_process,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count(),
        remove_columns=["text"],
    )
    print(sum(len(e["sentences"]) for e in datasets["train"]))
    # datasets.save_to_disk(args.output_dir)

    sentences = []
    for example in datasets["train"]:
        sentences += example["sentences"]

    (args.output_dir / "all.txt").write_text("\n".join(sentences))
    train_examples = random.sample(sentences, args.num_train_examples)
    (args.output_dir / "train.txt").write_text("\n".join(train_examples))


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
