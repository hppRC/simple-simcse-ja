import os
import random
from pathlib import Path

import deepl
from classopt import classopt
from more_itertools import chunked

from src import utils

DEEPL_AUTH_KEY = os.environ.get("DEEPL_AUTH_KEY")


@classopt(default_long=True)
class Args:
    data_dir: Path = "./data"
    save_dir: Path = "./data/nli-translated/deepl"
    batch_size: int = 16
    seed: int = 42


def main(args: Args):
    utils.set_seed(args.seed)
    translator = deepl.Translator(DEEPL_AUTH_KEY)
    if (args.save_dir / "translations.json").exists():
        translations: dict[str, str] = utils.load_json(args.save_dir / "translations.json")
    else:
        translations: dict[str, str] = {}

    def make_translations(batch: list[str]) -> dict[str, str]:
        result = translator.translate_text(
            batch,
            source_lang="en",
            target_lang="ja",
            preserve_formatting=True,
        )
        return {src: tgt.text for src, tgt in zip(batch, result)}

    target_sentences = []

    for name in ["train"]:
        # for name in ["train", "dev", "test"]:
        df = utils.load_jsonl(args.data_dir / "snli" / "snli_1.0" / f"snli_1.0_{name}.jsonl")
        # df = df.sample(n=1500)
        df = df.head(2000)
        target_sentences += df["sentence1"].tolist() + df["sentence2"].tolist()

    for name in ["train"]:
        # for name in ["train", "dev_matched", "dev_mismatched"]:
        df = utils.load_jsonl(
            args.data_dir / "mnli" / "multinli_1.0" / f"multinli_1.0_{name}.jsonl"
        )
        # df = df.sample(n=1500)
        df = df.head(2000)
        target_sentences += df["sentence1"].tolist() + df["sentence2"].tolist()

    target_sentences = list(sorted(set(target_sentences)))
    target_sentences = [s for s in target_sentences if s not in translations]
    # target_sentences = target_sentences[:5000]
    # target_sentences = random.sample(target_sentences, 5000)

    print(len(translations))
    print(len(target_sentences))
    print(sum(len(s) for s in target_sentences))

    for batch in chunked(target_sentences, args.batch_size):
        try:
            translations.update(make_translations(batch))
        except Exception as e:
            print(e)
            print(batch)

    utils.save_json(translations, args.save_dir / "translations.json")
    utils.save_config(args, args.save_dir / "config.json")

    for name in ["train", "dev", "test"]:
        df = utils.load_jsonl(args.data_dir / "snli" / "snli_1.0" / f"snli_1.0_{name}.jsonl")
        df = df[["sentence1", "sentence2", "gold_label"]]
        df["sentence1"] = df["sentence1"].map(translations)
        df["sentence2"] = df["sentence2"].map(translations)
        utils.save_jsonl(df, args.save_dir / "snli" / f"{name}.jsonl")

    for name in ["train", "dev_matched", "dev_mismatched"]:
        df = utils.load_jsonl(
            args.data_dir / "mnli" / "multinli_1.0" / f"multinli_1.0_{name}.jsonl"
        )
        df = df[["sentence1", "sentence2", "gold_label"]]
        df["sentence1"] = df["sentence1"].map(translations)
        df["sentence2"] = df["sentence2"].map(translations)
        utils.save_jsonl(df, args.save_dir / "mnli" / f"{name}.jsonl")


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
