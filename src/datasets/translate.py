from pathlib import Path

import torch
from classopt import classopt
from torch.utils.data import DataLoader
from transformers import (
    BatchEncoding,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    NllbTokenizerFast,
)

from src import utils


@classopt(default_long=True)
class Args:
    data_dir: Path = "./data"
    method: str = "nllb"
    batch_size: int = 8
    num_beams: int = 5
    max_seq_len: int = 512
    max_new_tokens: int = 512
    logging_interval: int = 1000

    dtype: utils.torch_dtype = "bf16"
    seed: int = 42
    device: str = "cuda:0"


@torch.inference_mode()
def main(args: Args):
    utils.set_seed(args.seed)

    save_dir = args.data_dir / "nli-translated" / args.method

    if "nllb" in args.method:
        model_name = "facebook/nllb-200-3.3B"
        model: M2M100ForConditionalGeneration = (
            M2M100ForConditionalGeneration.from_pretrained(model_name).eval().to(args.device)
        )
        tokenizer: NllbTokenizerFast = NllbTokenizerFast.from_pretrained(
            model_name,
            src_lang="eng_Latn",
            tgt_lang="jpn_Jpan",
        )
        forced_bos_token_id = tokenizer.lang_code_to_id["jpn_Jpan"]

    elif "m2m" in args.method:
        model_name = "facebook/m2m100-12B-last-ckpt"
        model: M2M100ForConditionalGeneration = (
            M2M100ForConditionalGeneration.from_pretrained(model_name).eval().to(args.device)
        )

        tokenizer: M2M100Tokenizer = M2M100Tokenizer.from_pretrained(
            model_name,
            src_lang="en",
            tgt_lang="ja",
        )
        forced_bos_token_id = tokenizer.lang_code_to_id["ja"]

    else:
        raise ValueError(f"Unknown model name: {args.method}")

    target_sentences = []
    for name in ["train", "dev", "test"]:
        df = utils.load_jsonl(args.data_dir / "snli" / "snli_1.0" / f"snli_1.0_{name}.jsonl")
        target_sentences += df["sentence1"].tolist() + df["sentence2"].tolist()

    for name in ["train", "dev_matched", "dev_mismatched"]:
        df = utils.load_jsonl(
            args.data_dir / "mnli" / "multinli_1.0" / f"multinli_1.0_{name}.jsonl"
        )
        target_sentences += df["sentence1"].tolist() + df["sentence2"].tolist()

    target_sentences = list(sorted(set(target_sentences)))

    def encode(batch: list[str]) -> BatchEncoding:
        return tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=args.max_seq_len,
        )

    dataloader = DataLoader(
        target_sentences,
        collate_fn=encode,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    generated_sentences = []
    for step, batch in enumerate(utils.tqdm(dataloader)):
        with torch.cuda.amp.autocast(dtype=args.dtype):
            generated_tokens = model.generate(
                **batch.to(args.device),
                num_beams=args.num_beams,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=args.max_new_tokens,
            )
        generated_sentences += tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        if step % args.logging_interval == 0 and len(generated_sentences) < 10000:
            translations = dict(zip(target_sentences, generated_sentences))
            utils.save_json(translations, save_dir / "translations.json")

    translations = dict(zip(target_sentences, generated_sentences))

    utils.save_json(translations, save_dir / "translations.json")
    utils.save_config(args, save_dir / "config.json")

    for name in ["train", "dev", "test"]:
        df = utils.load_jsonl(args.data_dir / "snli" / "snli_1.0" / f"snli_1.0_{name}.jsonl")
        df = df[["sentence1", "sentence2", "gold_label"]]
        df["sentence1"] = df["sentence1"].map(translations)
        df["sentence2"] = df["sentence2"].map(translations)
        utils.save_jsonl(df, save_dir / "snli" / f"{name}.jsonl")

    for name in ["train", "dev_matched", "dev_mismatched"]:
        df = utils.load_jsonl(
            args.data_dir / "mnli" / "multinli_1.0" / f"multinli_1.0_{name}.jsonl"
        )
        df = df[["sentence1", "sentence2", "gold_label"]]
        df["sentence1"] = df["sentence1"].map(translations)
        df["sentence2"] = df["sentence2"].map(translations)
        utils.save_jsonl(df, save_dir / "mnli" / f"{name}.jsonl")


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
