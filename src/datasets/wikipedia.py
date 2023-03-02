import os
import random
import re
from pathlib import Path

from classopt import classopt
from more_itertools import flatten

from datasets import DatasetDict, load_dataset
from src.datasets.common import preprocess_text


@classopt(default_long=True)
class Args:
    output_dir: Path = "./datasets/text/wikipedia"
    num_train_examples: int = 1_000_000
    seed: int = 42


def filter_text(text: str) -> bool:
    # filter out text containing equations
    return (not "\displaystyle" in text) and (len(text) > 1)


# 東北大BERTの事前学習用データ作成時の前処理を参考に作成
# see also: https://github.com/cl-tohoku/bert-japanese/blob/041b855db36942d42f2ca062da14416eaef93223/make_corpus_wiki.py#L54
def specific_preprocess(text: str, title: str = None) -> str:
    # remove templates
    text = re.sub(r"\[\d+?\]", "", text)
    text = re.sub(r"\[要.+?\]", "", text)
    text = re.sub(r"\{\{+[^{}]+?\}\}+", "", text)

    # remove navigation
    if title is not None:
        text = re.sub(r"^.+? \> " + re.escape(title), "", text)

    # remove footnotes
    text = re.sub(r" \^ .+", "", text)
    # remove annotations
    text = re.sub(r"\[(要出典|リンク切れ|.+?\?)\]", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def process(text: str, title: str = None):
    text: str = specific_preprocess(text, title)
    sentences: list[str] = [s.strip() for s in text.splitlines()]
    sentences: list[str] = list(flatten(preprocess_text(s) for s in sentences))
    sentences: list[str] = list(filter(filter_text, sentences))
    return {
        "title": title,
        "sentences": sentences,
    }


def batch_process(batch):
    examples = [process(text, title) for text, title in zip(batch["text"], batch["title"])]
    return {
        "title": [e["title"] for e in examples],
        "sentences": [e["sentences"] for e in examples],
    }


def main(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    datasets: DatasetDict = load_dataset(
        "wikipedia",
        language="ja",
        date="20230101",
        beam_runner="DirectRunner",
    )

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
