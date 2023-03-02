from itertools import count
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding

import src.utils as utils
from src.experiment import CommonArgs, UnsupSimCSEExperiment


class Args(CommonArgs):
    METHOD = "unsup-simcse"

    model_name: str = "studio-ousia/luke-japanese-base-lite"
    dataset_dir: Path = "datasets/text"
    dataset_name: str = "wiki40b"

    batch_size: int = 128
    lr: float = 3e-5


def main(args: Args):
    utils.set_seed(args.seed)

    exp = UnsupSimCSEExperiment(args)

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
                    emb1 = exp.model.forward(**batch)
                    emb2 = exp.model.forward(**batch)

                sim_mat = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)
                sim_mat = sim_mat / args.temperature

                labels = torch.arange(sim_mat.size(0)).long().to(args.device)
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

    exp.model.load_state_dict(best_state_dict)
    exp.model.eval().to(args.device)

    sts_metrics = exp.sts_test()
    utils.save_json(sts_metrics, args.output_dir / "sts-metrics.json")
    utils.save_config(args, args.output_dir / "config.json")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
