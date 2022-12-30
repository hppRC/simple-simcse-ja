import random
from pathlib import Path

from torch.utils.data import Dataset

import src.utils as utils


class UnsupSimCSEDataset(Dataset):
    def __init__(
        self,
        path: Path,
    ):
        self.path: Path = path
        self.data: list[str] = path.read_text().splitlines()

    def __getitem__(self, index: int) -> str:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class SupSimCSEDataset(Dataset):
    def __init__(
        self,
        path: Path,
    ):
        self.path: Path = path
        self.data: list[dict] = utils.load_jsonl(path).to_dict("records")

    def __getitem__(self, index: int) -> tuple[str, str, str]:
        example: dict = self.data[index]
        sent0: str = random.choice(example["sent0"])
        sent1: str = random.choice(example["sent1"])

        if len(example["hard_neg"]) > 0:
            hard_neg: str = random.choice(example["hard_neg"])
        else:
            while True:
                random_example = random.choice(self.data)
                if random_example["id"] != example["id"]:
                    break
            hard_neg: str = random.choice(random_example["sent1"])
        return sent0, sent1, hard_neg

    def __len__(self) -> int:
        return len(self.data)
