import random
import unicodedata
from collections import defaultdict
from pathlib import Path

from classopt import classopt

from src import utils


@classopt(default_long=True)
class Args:
    dataset_dir: Path = "./datasets"
    save_dir: Path = "./datasets/nli/jsnli+wiki40b-010"
    unsup_examples_ratio: float = 0.1
    seed: int = 42


def normalize(text: str) -> str:
    text = text.replace(" ", "").replace("。", "").replace("、", "").strip()
    text = unicodedata.normalize("NFKC", text)
    return text


def main(args: Args):
    utils.set_seed(args.seed)

    nli_all = utils.load_jsonl(args.dataset_dir / "nli/jsnli/train.jsonl").to_dict("records")
    wiki_all: list[str] = (args.dataset_dir / "wiki40b/train.txt").read_text().splitlines()

    examples = defaultdict(lambda: defaultdict(set))

    for sentence in wiki_all:
        identifier = normalize(sentence)
        examples[identifier]["sent0"] |= set([sentence])
        examples[identifier]["sent1"] |= set([sentence])

    wiki_all = []
    for example_dict in examples.values():
        sent0: list[str] = list(example_dict["sent0"])
        sent1: list[str] = list(example_dict["sent1"])
        hard_neg: list[str] = list(example_dict["hard_neg"])
        if len(sent0) == 0 or len(sent1) == 0:
            continue
        wiki_all.append({"sent0": sent0, "sent1": sent1, "hard_neg": hard_neg})

    # if args.num_unsup_examples <= len(wiki_all):
    #     wiki_all = random.sample(wiki_all, args.num_unsup_examples)
    # wiki_all = random.sample(wiki_all, args.num_total_examples - len(nli_all))

    wiki_all = random.sample(wiki_all, int(len(nli_all) * args.unsup_examples_ratio))

    data = nli_all + wiki_all
    random.shuffle(data)
    data = [{**example, "id": idx} for idx, example in enumerate(data)]
    print(len(data), len(wiki_all), len(nli_all))

    utils.save_jsonl(data, args.save_dir / "train.jsonl")


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
