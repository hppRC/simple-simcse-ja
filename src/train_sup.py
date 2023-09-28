from itertools import count
from pathlib import Path

import src.utils as utils
import torch
import torch.nn.functional as F
from src.experiment import CommonArgs, SupSimCSEExperiment
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding


class Args(CommonArgs):
    METHOD = "sup-simcse"

    model_name: str = "studio-ousia/luke-japanese-base-lite"
    dataset_dir: Path = "datasets/nli"
    dataset_name: str = "jsnli"

    batch_size: int = 512
    lr: float = 1e-5


def main(args: Args):
    utils.set_seed(args.seed)

    exp = SupSimCSEExperiment(args)

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

                labels = (
                    torch.arange(sim_mat.size(0))
                    .long()
                    .to(args.device, non_blocking=True)
                )
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
