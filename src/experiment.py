from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

import src.utils as utils
from src.dataset import SupSimCSEDataset, UnsupSimCSEDataset
from src.models import SimCSEModel
from src.sts import STSEvaluation


class CommonArgs(Tap):
    METHOD: str = None

    model_name: str
    dataset_dir: Path
    dataset_name: str

    sts_dir: Path = "./datasets/sts"

    batch_size: int
    lr: float

    temperature: float = 0.05
    mlp_only_train: bool = True
    max_seq_len: int = 64

    use_jumanpp: bool = False

    num_training_examples: int = 2**20
    num_eval_logging: int = 2**6
    num_warmup_ratio: float = 0.1

    seed: int = None
    device: str = "cuda:0"
    dtype: utils.torch_dtype = "bf16"

    def process_args(self):
        if self.METHOD is None:
            raise ValueError("METHOD is not defined.")

        date, time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S").split("/")

        self.output_dir = self.make_output_dir(
            "outputs",
            self.METHOD,
            self.dataset_name,
            self.model_name,
            date,
            time,
        )
        self.num_training_steps: int = self.num_training_examples // self.batch_size
        self.num_warmup_steps: int = int(self.num_training_steps * self.num_warmup_ratio)
        self.eval_logging_interval: int = (self.num_training_steps - 1) // self.num_eval_logging + 1

    def make_output_dir(self, *args) -> Path:
        args = [str(a).replace("/", "__") for a in args]
        output_dir = Path(*args)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @property
    def data_dir(self) -> Path:
        return self.dataset_dir / self.dataset_name


@dataclass
class Experiment:
    args: CommonArgs

    def __post_init__(self):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            model_max_length=self.args.max_seq_len,
            use_fast=False,
        )
        self.model: SimCSEModel = (
            SimCSEModel(
                model_name=self.args.model_name,
                mlp_only_train=self.args.mlp_only_train,
            )
            .eval()
            .to(self.args.device)
        )

        self.sts = STSEvaluation(
            sts_dir=self.args.sts_dir,
            do_jumanpp_preprocess=self.args.use_jumanpp,
        )

    def tokenize(self, batch: list[str]) -> BatchEncoding:
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.max_seq_len,
        )

    def create_loader(
        self,
        dataset: list[str] | list[dict],
        collate_fn: callable = None,
        batch_size: int = None,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            collate_fn=collate_fn or self.collate_fn,
            batch_size=batch_size or self.args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=drop_last,
        )

    def collate_fn(self, data_list: list[dict]) -> BatchEncoding:
        raise NotImplementedError

    def encode_collate_fn(self, data_list: list[str]) -> BatchEncoding:
        return self.tokenize(data_list)

    @torch.inference_mode()
    def encode(
        self,
        sentences: list[str],
        batch_size: int = None,
        convert_to_numpy: bool = False,
        **_,
    ) -> torch.Tensor | np.ndarray:
        self.model.eval()
        data_loader = self.create_loader(
            sentences,
            collate_fn=self.encode_collate_fn,
            batch_size=batch_size or self.args.batch_size,
        )

        embs = []
        for batch in data_loader:
            with torch.cuda.amp.autocast(dtype=self.args.dtype):
                emb = self.model.forward(**batch.to(self.args.device))
            embs.append(emb.cpu())
        embs = torch.cat(embs, dim=0)

        if convert_to_numpy:
            embs: np.ndarray = embs.numpy()
        return embs

    def sts_dev(self):
        self.model.eval()
        return self.sts.dev(encode=self.encode)

    def sts_test(self):
        self.model.eval()
        return self.sts(encode=self.encode)

    def clone_state_dict(self) -> dict:
        return {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}

    def log(self, metrics: dict) -> None:
        utils.log(metrics, self.args.output_dir / "log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']} \t"
            f"step: {metrics['step']} \t"
            f"loss: {metrics['loss']:2.6f}       \t"
            f"sts-dev: {metrics['sts-dev']:.4f}"
        )


@dataclass
class UnsupSimCSEExperiment(Experiment):
    def __post_init__(self):
        super().__post_init__()

        self.train_dataset = UnsupSimCSEDataset(
            self.args.data_dir / "train.txt",
            do_jumanpp_preprocess=self.args.use_jumanpp,
        )

        self.train_dataloader = self.create_loader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
        )

    def collate_fn(self, data_list: list[str]) -> BatchEncoding:
        return self.tokenize(data_list)


@dataclass
class SupSimCSEExperiment(Experiment):
    def __post_init__(self):
        super().__post_init__()

        self.train_dataset = SupSimCSEDataset(
            self.args.data_dir / "train.jsonl",
            do_jumanpp_preprocess=self.args.use_jumanpp,
        )

        self.train_dataloader = self.create_loader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
        )

    def collate_fn(self, data_list: list[dict]) -> BatchEncoding:
        sent0, sent1, hard_neg = zip(*data_list)
        return BatchEncoding(
            {
                "sent0": self.tokenize(sent0),
                "sent1": self.tokenize(sent1),
                "hard_neg": self.tokenize(hard_neg),
            }
        )
