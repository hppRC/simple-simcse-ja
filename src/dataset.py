import random
from pathlib import Path

from torch.utils.data import Dataset as TorchDataset

from datasets import Dataset as HFDataset
from datasets import load_dataset
from src import utils


def jumanpp_unsup_process(example: dict) -> dict:
    return {
        "text": utils.jumanpp_wakati(example["text"]),
    }


class UnsupSimCSEDataset(TorchDataset):
    def __init__(
        self,
        path: Path,
        do_jumanpp_preprocess: bool = False,
    ):
        self.path: Path = path
        dataset: HFDataset = load_dataset("text", data_files=str(path), split="train")

        if do_jumanpp_preprocess:
            dataset = dataset.map(
                jumanpp_unsup_process,
                # Jumanppを使うとマルチプロセス処理ができないので、num_proc=1にしてシングルプロセスで実行するように設定
                num_proc=1,
                remove_columns=["text"],
            )

        self.data = dataset["text"]

    def __getitem__(self, index: int) -> str:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def jumanpp_sup_process(example: dict[str, list[str]]) -> dict:
    sent0 = [utils.jumanpp_wakati(s) for s in example["sent0"]]
    sent1 = [utils.jumanpp_wakati(s) for s in example["sent1"]]
    hard_neg = [utils.jumanpp_wakati(s) for s in example["hard_neg"]]

    return {
        "sent0": sent0,
        "sent1": sent1,
        "hard_neg": hard_neg,
    }


class SupSimCSEDataset(TorchDataset):
    def __init__(
        self,
        path: Path,
        do_jumanpp_preprocess: bool = False,
    ):
        self.path: Path = path
        dataset: HFDataset = load_dataset("json", data_files=str(path), split="train")

        if do_jumanpp_preprocess:
            dataset = dataset.map(
                jumanpp_sup_process,
                num_proc=1,
            )

        self.data = list(dataset)

    def __getitem__(self, index: int) -> tuple[str, str, str]:
        example: dict[str, list[str]] = self.data[index]
        sent0: str = random.choice(example["sent0"])
        sent1: str = random.choice(example["sent1"])

        if len(example["hard_neg"]) > 0:
            hard_neg: str = random.choice(example["hard_neg"])
        else:
            random_example = random.choice(self.data)
            while random_example["id"] == example["id"]:
                random_example = random.choice(self.data)

            hard_neg: str = random.choice(random_example["sent1"])
        return sent0, sent1, hard_neg

    def __len__(self) -> int:
        return len(self.data)
