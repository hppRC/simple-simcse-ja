import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from classopt import classopt
from mteb import MTEB
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, logging
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

from src.sts import STSEvaluation

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@classopt(default_long=True)
class Args:
    method: str = "unsup-simcse"
    model_name: str = "cl-tohoku/bert-base-japanese-v2"

    dataset_dir: Path = "./datasets"
    dataset_name: str = "wiki40b"
    sts_dir: Path = "./datasets/sts"

    num_train: int = 1_000_000

    batch_size: int = 128
    epochs: int = 1
    lr: float = 2e-5
    num_warmup_steps: int = 0
    temperature: float = 0.05

    max_seq_len: int = 128

    eval_logging_interval: int = 50
    not_amp: bool = False
    device: str = "cuda:0"
    seed: int = 42

    def __post_init__(self):
        date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        model_name = self.model_name.replace("/", "__")
        self.output_dir = Path("outputs") / self.method / self.dataset_name / model_name / date
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Experiment:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    sts: STSEvaluation
    method: str
    batch_size: int

    def tokenize(self, batch: List[str]) -> BatchEncoding:
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=args.max_seq_len,
        )

    def unsup_simcse_collate_fn(self, data_list: List[str]) -> BatchEncoding:
        return self.tokenize(data_list)

    def sup_simcse_collate_fn(self, data_list: List[Tuple[str, str]]) -> BatchEncoding:
        premise, hypothesis = zip(*data_list)
        return BatchEncoding(
            {
                "premise": self.tokenize(premise),
                "hypothesis": self.tokenize(hypothesis),
            }
        )

    def create_loader(
        self,
        dataset: List[str],
        batch_size: int = None,
        shuffle: bool = False,
        drop_last: bool = False,
        for_evaluation: bool = False,
    ) -> DataLoader:
        if for_evaluation or self.method == "unsup-simcse":
            collate_fn = self.unsup_simcse_collate_fn
        elif self.method == "sup-simcse":
            collate_fn = self.sup_simcse_collate_fn
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=drop_last,
        )

    @torch.inference_mode()
    def encode(
        self,
        sentences: List[str],
        batch_size: int = None,
        convert_to_numpy: bool = False,
        **_,
    ) -> torch.Tensor:
        embs = []
        self.model.eval()
        for batch in self.create_loader(
            sentences,
            batch_size=batch_size or self.batch_size,
            for_evaluation=True,
        ):
            emb = self.model.forward(**batch.to(args.device))
            embs.append(emb.cpu())
        embs = torch.cat(embs, dim=0)
        if convert_to_numpy:
            embs = embs.numpy()
        return embs

    def sts_dev(self):
        self.model.eval()
        return self.sts.dev(encode=self.encode)

    def sts_test(self):
        self.model.eval()
        return self.sts(encode=self.encode)


class SimCSEDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_name: str,
        method: str,
        num_train: int,
    ):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.method = method
        self.num_train = num_train
        self.dir = dataset_dir / dataset_name

        if self.method == "unsup-simcse":
            self.path = self.dir / "train.txt"
            self.data: List[str] = self.path.read_text().splitlines()
            self.data: List[str] = random.sample(self.data, num_train)

        elif self.method == "sup-simcse":
            if self.dataset_name == "janli":
                self.path = self.dir / "janli.tsv"
                df = pd.read_table(self.path)
                df = df[df["entailment_label_Ja"] == "entailment"]
                df = df[df["split"] == "train"]
                premise = df["sentence_A_Ja"].tolist()
                hypothesis = df["sentence_B_Ja"].tolist()
                self.data: List[Tuple[str, str]] = list(zip(premise, hypothesis))

            elif self.dataset_name == "jnli":
                self.path = self.dir / "train-v1.1.json"
                df = pd.read_json(self.path, orient="records", lines=True)
                df = df[df["label"] == "entailment"]
                premise = df["sentence1"].tolist()
                hypothesis = df["sentence2"].tolist()
                self.data: List[Tuple[str, str]] = list(zip(premise, hypothesis))

            elif self.dataset_name == "jsnli":
                self.path = self.dir / "train_w_filtering.tsv"
                df = pd.read_table(self.path, header=None, names=["label", "premise", "hypothesis"])
                df = df[df["label"] == "entailment"]
                premise = [s.replace(" ", "") for s in df["premise"].tolist()]
                hypothesis = [s.replace(" ", "") for s in df["hypothesis"].tolist()]
                self.data: List[Tuple[str, str]] = list(zip(premise, hypothesis))
            else:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class SimCSEModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone: PreTrainedModel = AutoModel.from_pretrained(model_name)
        self.hidden_size: int = self.backbone.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor = None,
        token_type_ids: Tensor = None,
    ) -> Tensor:
        outputs: BaseModelOutput = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        emb = outputs.last_hidden_state[:, 0]
        emb = self.dense(emb)
        emb = self.activation(emb)
        return emb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args: Args):
    logging.set_verbosity_error()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model: SimCSEModel = SimCSEModel(args.model_name).eval().to(args.device)

    exp = Experiment(
        model=model,
        tokenizer=tokenizer,
        sts=STSEvaluation(args.sts_dir),
        method=args.method,
        batch_size=args.batch_size,
    )

    train_dataset = SimCSEDataset(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        method=args.method,
        num_train=args.num_train,
    )
    train_dataloader = exp.create_loader(train_dataset, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=len(train_dataloader) * args.epochs,
    )

    if args.method == "unsup-simcse":

        def train_step(batch: BatchEncoding) -> float:
            emb1 = model.forward(**batch)
            emb2 = model.forward(**batch)

            sim_matrix = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)
            sim_matrix = sim_matrix / args.temperature

            labels = torch.arange(args.batch_size).long().to(args.device)
            return F.cross_entropy(sim_matrix, labels)

    elif args.method == "sup-simcse":

        def train_step(batch: BatchEncoding) -> float:
            pre = model.forward(**batch.premise)
            hyp = model.forward(**batch.hypothesis)

            sim_matrix = F.cosine_similarity(pre.unsqueeze(1), hyp.unsqueeze(0), dim=-1)
            sim_matrix = sim_matrix / args.temperature

            labels = torch.arange(args.batch_size).long().to(args.device)
            return F.cross_entropy(sim_matrix, labels)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    scaler = torch.cuda.amp.GradScaler(enabled=not args.not_amp)

    best_stsb = exp.sts_dev()
    best_epoch, best_step = 0, 0
    best_state_dict = model.state_dict()

    print(f"epoch: {0:>3} |\tstep: {0:>6} |\tloss: {' '*9}nan |\tJSICK train: {best_stsb:.4f}")
    logs: List[Dict[str, Union[int, float]]] = [
        {
            "epoch": 0,
            "step": best_step,
            "loss": None,
            "stsb": best_stsb,
        }
    ]

    for epoch in range(args.epochs):
        model.train()

        for step, batch in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            dynamic_ncols=True,
            leave=False,
        ):
            with torch.cuda.amp.autocast(enabled=not args.not_amp):
                batch: BatchEncoding = batch.to(args.device)
                loss = train_step(batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scale = scaler.get_scale()
            scaler.update()
            if scale <= scaler.get_scale():
                lr_scheduler.step()

            total_steps = len(train_dataloader) * epoch + step + 1
            if (total_steps % args.eval_logging_interval == 0) or (
                total_steps == len(train_dataloader) * args.epochs
            ):
                model.eval()
                stsb_score = exp.sts_dev()

                if best_stsb < stsb_score:
                    best_stsb = stsb_score
                    best_epoch, best_step = epoch, total_steps
                    best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

                tqdm.write(
                    f"epoch: {epoch:>3} |\tstep: {step+1:>6} |\tloss: {loss.item():.10f} |\tJSICK train: {stsb_score:.4f}"
                )
                logs.append(
                    {
                        "epoch": epoch,
                        "step": step + 1,
                        "loss": loss.item(),
                        "stsb": stsb_score,
                    }
                )
                pd.DataFrame(logs).to_csv(args.output_dir / "logs.csv", index=False)
                model.train()

    with (args.output_dir / "dev-metrics.json").open("w") as f:
        data = {
            "best-epoch": best_epoch,
            "best-step": best_step,
            "best-stsb": best_stsb,
        }
        json.dump(data, f, indent=2, ensure_ascii=False)

    model.load_state_dict(best_state_dict)
    model.eval().to(args.device)

    sts_metrics = exp.sts_test()
    with (args.output_dir / "sts-metrics.json").open("w") as f:
        json.dump(sts_metrics, f, indent=2, ensure_ascii=False)

    with (args.output_dir / "config.json").open("w") as f:
        data = {k: v if type(v) in [int, float] else str(v) for k, v in vars(args).items()}
        json.dump(data, f, indent=2, ensure_ascii=False)

    # metb = MTEB(task_langs=["ja"])
    # metb.run(exp, output_folder=args.output_dir / "mteb")


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
