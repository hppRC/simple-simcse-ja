from datetime import datetime
from itertools import count
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from classopt import classopt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

import src.utils as utils
from src.dataset import SupSimCSEDataset
from src.models import SimCSEModel
from src.sts import STSEvaluation


@classopt(default_long=True)
class Args:
    method: str = "sup-simcse"
    model_name: str = "studio-ousia/luke-japanese-base-lite"

    dataset_dir: Path = "./datasets/nli"
    dataset_name: str = "jsnli"
    sts_dir: Path = "./datasets/sts"

    batch_size: int = 512
    num_training_examples: int = 2**18
    num_eval_logging: int = 16
    lr: float = 1e-5
    num_warmup_ratio: float = 0.1

    mlp_type: str = "simcse"
    temperature: float = 0.05
    mlp_only_train: bool = True
    max_seq_len: int = 128

    not_amp: bool = False
    device: str = "cuda:0"
    dtype: utils.torch_dtype = "bf16"
    seed: int = 42

    def __post_init__(self):
        utils.set_seed(self.seed)

        date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        model_name = self.model_name.replace("/", "__")
        self.output_dir = (
            Path("outputs") / self.method / self.dataset_name / model_name / self.mlp_type / date
        )

        self.num_training_steps: int = self.num_training_examples // self.batch_size
        self.num_warmup_steps: int = int(self.num_training_steps * self.num_warmup_ratio)
        self.eval_logging_interval: int = (self.num_training_steps - 1) // self.num_eval_logging + 1


class Experiment:
    args: Args
    model: PreTrainedModel = None
    tokenizer: PreTrainedTokenizer = None
    sts: STSEvaluation = None

    def __init__(self, args: Args) -> None:
        self.args = args

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            model_max_length=args.max_seq_len,
        )
        self.model: SimCSEModel = (
            SimCSEModel(
                model_name=args.model_name,
                mlp_type=args.mlp_type,
                mlp_only_train=args.mlp_only_train,
            )
            .eval()
            .to(args.device, non_blocking=True)
        )
        self.sts = STSEvaluation(args.sts_dir)

    def tokenize(self, batch: list[str]) -> BatchEncoding:
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.max_seq_len,
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

    def create_loader(
        self,
        dataset: list[str] | list[dict],
        batch_size: int = None,
        shuffle: bool = False,
        drop_last: bool = False,
        for_evaluation: bool = False,
    ) -> DataLoader:
        if for_evaluation:
            collate_fn = self.tokenize
        else:
            collate_fn = self.collate_fn

        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size or self.args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=drop_last,
        )

    @torch.inference_mode()
    def encode(
        self,
        sentences: list[str],
        batch_size: int = None,
        convert_to_numpy: bool = False,
        **_,
    ) -> torch.Tensor | np.ndarray:
        self.model.eval()

        embs = []
        for batch in self.create_loader(
            sentences,
            batch_size=batch_size or self.args.batch_size,
            for_evaluation=True,
        ):
            with torch.cuda.amp.autocast(enabled=not self.args.not_amp, dtype=self.args.dtype):
                emb = self.model.forward(**batch.to(self.args.device))
            embs.append(emb.float().cpu())
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


def main(args: Args) -> float:
    exp = Experiment(args)

    train_dataset = SupSimCSEDataset(path=args.dataset_dir / args.dataset_name / "train.jsonl")
    train_dataloader = exp.create_loader(train_dataset, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(params=exp.model.parameters(), lr=args.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=args.num_training_steps,
        num_warmup_steps=args.num_warmup_steps,
    )

    best_dev_score = exp.sts_dev()
    best_epoch, best_step = 0, 0
    best_state_dict = exp.clone_state_dict()

    print(f"epoch: {0:>3} |\tstep: {0:>6} |\tloss: {' '*9}nan |\tJSICK train: {best_dev_score:.4f}")
    utils.log(
        {
            "epoch": best_epoch,
            "step": best_step,
            "loss": None,
            "stsb": best_dev_score,
        },
        args.output_dir / "logs.csv",
    )

    scaler = torch.cuda.amp.GradScaler(enabled=not args.not_amp)

    with utils.ProgressBar(n=args.num_training_steps) as pbar:
        current_step = 0

        for epoch in count():
            exp.model.train()

            for batch in train_dataloader:
                batch = batch.to(args.device)
                with torch.cuda.amp.autocast(enabled=not args.not_amp, dtype=args.dtype):
                    sent0 = exp.model.forward(**batch.sent0)
                    sent1 = exp.model.forward(**batch.sent1)
                    hard_neg = exp.model.forward(**batch.hard_neg)

                    sim_mat_1st = F.cosine_similarity(
                        sent0.unsqueeze(1), sent1.unsqueeze(0), dim=-1
                    )
                    sim_mat_2nd = F.cosine_similarity(
                        sent0.unsqueeze(1), hard_neg.unsqueeze(0), dim=-1
                    )

                    sim_mat = torch.cat([sim_mat_1st, sim_mat_2nd], dim=1)
                    sim_mat = sim_mat / args.temperature

                    labels = torch.arange(sim_mat.size(0)).long().to(args.device, non_blocking=True)
                    loss = F.cross_entropy(sim_mat, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scale = scaler.get_scale()
                scaler.update()
                # to avoid warnings
                if scale <= scaler.get_scale():
                    lr_scheduler.step()

                current_step += 1

                if (current_step % args.eval_logging_interval == 0) or (
                    current_step == args.num_training_steps
                ):
                    exp.model.eval()
                    dev_score = exp.sts_dev()

                    if best_dev_score < dev_score:
                        best_dev_score = dev_score
                        best_epoch, best_step = epoch, current_step
                        best_state_dict = exp.clone_state_dict()

                    tqdm.write(
                        f"epoch: {epoch:>3} |\tstep: {current_step:>6} |\tloss: {loss.item():.10f} |\tJSICK train: {dev_score:.4f}"
                    )
                    utils.log(
                        {
                            "epoch": epoch,
                            "step": current_step,
                            "loss": loss.item(),
                            "stsb": dev_score,
                        },
                        args.output_dir / "logs.csv",
                    )
                    exp.model.train()

                pbar.update()
                if current_step == args.num_training_steps:
                    break

            if current_step == args.num_training_steps:
                break

    dev_metrics = {
        "best-epoch": best_epoch,
        "best-step": best_step,
        "best-dev": best_dev_score,
    }
    utils.save_json(dev_metrics, args.output_dir / "dev-metrics.json")

    exp.model.load_state_dict(best_state_dict)
    exp.model.eval().to(args.device, non_blocking=True)

    sts_metrics = exp.sts_test()
    utils.save_json(sts_metrics, args.output_dir / "sts-metrics.json")
    utils.save_config(args, args.output_dir / "config.json")

    print(sts_metrics)

    # metb = MTEB(task_langs=["ja"])
    # metb.run(exp, output_folder=args.output_dir / "mteb")
    return best_dev_score


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
