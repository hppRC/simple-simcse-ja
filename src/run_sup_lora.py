from dataclasses import dataclass
from itertools import count, product
from pathlib import Path

import peft
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    MT5EncoderModel,
    PretrainedConfig,
    PreTrainedTokenizer,
    T5EncoderModel,
)
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding

import src.utils as utils
from src import utils
from src.experiment import CommonArgs, SupSimCSEExperiment


class Args(CommonArgs):
    METHOD = "lora-sup-simcse"

    model_name: str = "google/mt5-large"
    dataset_dir: Path = "datasets/nli"
    dataset_name: str = "jsnli"
    use_lora: bool = True
    pooling: str = "mean"
    mlp_only_train: bool = True
    gradient_checkpointing: bool = True


@dataclass
class Config:
    lora_r: int
    lr: float
    batch_size: int

    LORA_R = [16, 8, 4]
    LEARNING_RATES = [5e-4, 3e-4, 1e-4]
    BATCH_SIZES = [256, 128, 64]

    @classmethod
    def all(cls):
        for lora_r, batch_size, lr in product(
            cls.LORA_R,
            cls.BATCH_SIZES,
            cls.LEARNING_RATES,
        ):
            yield cls(lora_r, lr, batch_size)


class Runner:
    def __init__(self, args: Args):
        self.args = args
        self.base_model_config: PretrainedConfig = AutoConfig.from_pretrained(args.model_name)
        if self.base_model_config.model_type == "mt5":
            self.base_model = MT5EncoderModel.from_pretrained(args.model_name)
        elif self.base_model_config.model_type == "t5":
            self.base_model = T5EncoderModel.from_pretrained(args.model_name)
        else:
            self.base_model = AutoModel.from_pretrained(args.model_name)
        self.base_model = self.base_model.to(args.device).eval()

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            model_max_length=self.args.max_seq_len,
            use_fast=False,
        )

    def write_trainable_params(self, model: PeftModel) -> None:
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        percentage = 100 * trainable_params / all_param
        all_param /= 1000000000
        trainable_params /= 1_000_000

        tqdm.write(
            f"trainable params: {trainable_params:.2f}M || "
            f"all params: {all_param:.2f}B || "
            f"trainable%: {percentage:.4f}"
        )

    def run_all(self):
        for config in Config.all():
            self.run(config)
            self.args.reset_output_dir()

    def run(self, config: Config):
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=32,
            lora_dropout=0.1,
            inference_mode=False,
        )
        model = PeftModel(self.base_model, peft_config)
        self.write_trainable_params(model)

        args.lr = config.lr
        args.batch_size = config.batch_size
        args.lora_r = config.lora_r

        exp = SupSimCSEExperiment(args=self.args, model=model)
        optimizer = torch.optim.AdamW(params=exp.model.parameters(), lr=args.lr)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=args.num_training_steps,
            num_warmup_steps=args.num_warmup_steps,
        )

        best_dev_score = exp.sts_dev()
        best_epoch, best_step = 0, 0
        val_metrics = {
            "epoch": best_epoch,
            "step": best_step,
            "loss": float("inf"),
            "sts-dev": best_dev_score,
        }
        exp.log(val_metrics)
        best_state_dict = exp.clone_state_dict()

        scaler = torch.cuda.amp.GradScaler()
        with tqdm(
            total=args.num_training_steps,
            dynamic_ncols=True,
            desc="Training",
        ) as pbar:
            current_step = 0
            train_losses = []

            for epoch in count():
                exp.model.train()

                for batch in tqdm(
                    exp.train_dataloader,
                    total=len(exp.train_dataloader),
                    dynamic_ncols=True,
                    leave=False,
                    desc="Epoch",
                ):
                    batch: BatchEncoding = batch.to(args.device)
                    with torch.cuda.amp.autocast(dtype=args.dtype):
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
                    train_losses.append(float(loss.item()))

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

                        val_metrics = {
                            "epoch": epoch,
                            "step": current_step,
                            "loss": sum(train_losses) / len(train_losses),
                            "sts-dev": dev_score,
                        }
                        exp.log(val_metrics)
                        train_losses = []

                        exp.model.train()

                    pbar.update()

                    if current_step == args.num_training_steps:
                        break
                else:
                    continue
                break

        dev_metrics = {
            "best-epoch": best_epoch,
            "best-step": best_step,
            "best-dev": best_dev_score,
        }
        utils.save_json(dev_metrics, args.output_dir / "dev-metrics.json")

        peft.set_peft_model_state_dict(exp.model.backbone, best_state_dict)
        exp.model.eval().to(args.device)

        sts_metrics = exp.sts_test()
        utils.save_json(sts_metrics, args.output_dir / "sts-metrics.json")
        utils.save_config(args, args.output_dir / "config.json")


def main(args: Args):
    runner = Runner(args)
    runner.run_all()


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
