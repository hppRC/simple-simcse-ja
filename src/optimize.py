from pathlib import Path

import optuna
from classopt import classopt
from optuna import Study, Trial

from src.train_sup import Args, main


@classopt(default_long=True)
class Opt:
    method: str = "sup-simcse"
    model_name: str = "studio-ousia/luke-japanese-base-lite"
    mlp_type: str = "simcse"
    dataset_dir: Path = "./datasets/nli"
    dataset_name: str = "jsnli"

    max_batch_size: int = 512

    n_trials: int = None
    timeout: int = None

    seed: int = None
    dtype: str = "bf16"
    device: str = "cuda:0"

    def __post_init__(self):
        if self.n_trials is None and self.timeout is None:
            self.n_trials = 1

        model_name = self.model_name.replace("/", "__")
        self.study_name = f"{self.method}/{self.dataset_name}/{model_name}/{self.mlp_type}"

        self.output_dir = Path(f"outputs/optuna/{self.study_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.storage_name = f"sqlite:///{str(self.output_dir)}/storage.db"


class Objective:
    def __init__(self, opt: Opt):
        self.opt = opt

    def __call__(self, trial: Trial):
        batch_size = trial.suggest_int("batch_size", 32, self.opt.max_batch_size, log=True)
        lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
        temperature = trial.suggest_float("temperature", 1e-5, 1.0, log=True)

        args = Args.from_dict(
            {
                **self.opt.to_dict(),
                "batch_size": batch_size,
                "lr": lr,
                "temperature": temperature,
            }
        )
        print(batch_size, lr, temperature)
        score = main(args)
        return score


def run():
    opt: Opt = Opt.from_args()

    study: Study = optuna.create_study(
        study_name=opt.study_name,
        storage=opt.storage_name,
        direction="maximize",
        load_if_exists=True,
    )

    objective = Objective(opt=opt)

    study.optimize(
        objective,
        n_trials=opt.n_trials,
        timeout=opt.timeout,
        catch=(RuntimeError,),
    )

    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)
    print("Best Trial: ", study.best_trial)


if __name__ == "__main__":
    run()
