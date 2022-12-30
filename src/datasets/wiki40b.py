import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Union

from classopt import classopt
from more_itertools import divide
from tqdm import tqdm

from datasets import Dataset, DatasetDict, load_dataset


@classopt(default_long=True)
class Args:
    output_dir: Path = "./datasets/wiki40b"


# entries of Wiki-40B are saved as a list of articles
# each article is a raw text, which have some special tags
# for convinience, we preprocess the raw text into a list of sentences
def process_article(raw_article: str) -> Dict[str, Union[str, List[str]]]:
    lines = [line.strip() for line in raw_article.strip().split("\n")]

    assert lines[0] == "_START_ARTICLE_"
    title = lines[1]
    lines = lines[2:]

    sentences: List[str] = []
    line_id = 0
    while line_id < len(lines):
        if lines[line_id] == "_START_SECTION_":
            line_id += 2
        elif lines[line_id] == "_START_PARAGRAPH_":
            line_id += 1
        else:
            ss: List[str] = [s.strip() for s in lines[line_id].split("_NEWLINE_")]
            ss: List[str] = [s for s in ss if s != ""]
            sentences += ss
            line_id += 1
    return {
        "title": title,
        "sentences": sentences,
    }


def process_chunk(chunk: List[str]) -> List[str]:
    return [process_article(raw_article) for raw_article in chunk]


def preprocess_and_save(
    dataset: Dataset,
    output_path: Path,
):
    articles = dataset["text"]
    sentences = []
    num_procs = os.cpu_count()

    # do preprocessing in parallel
    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        # process list of articles for each process
        chunks = divide(n=num_procs * 8, iterable=articles)
        for chunk in tqdm(
            executor.map(process_chunk, chunks),
            total=len(chunks),
        ):
            for data in chunk:
                sentences += data["sentences"]
    output_path.write_text("\n".join(sentences))


def main(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset: DatasetDict = load_dataset("wiki40b", "ja", beam_runner="DirectRunner")

    preprocess_and_save(dataset["train"], args.output_dir / "train.txt")
    preprocess_and_save(dataset["validation"], args.output_dir / "val.txt")
    preprocess_and_save(dataset["test"], args.output_dir / "test.txt")


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
