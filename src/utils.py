import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_jsonl(data: Union[Iterable, pd.DataFrame], path: Union[Path, str]) -> None:
    path = Path(path)

    if type(data) != pd.DataFrame:
        data = pd.DataFrame(data)
    data.to_json(
        path,
        orient="records",
        lines=True,
        force_ascii=False,
    )


def save_json(data: Dict[Any, Any], path: Union[Path, str]) -> None:
    path = Path(path)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl(
    path: Union[Path, str],
    as_pd: bool = False,
) -> Union[pd.DataFrame, List[Dict]]:
    path = Path(path)
    df = pd.read_json(path, lines=True)
    if as_pd:
        return df
    else:
        return df.to_dict(orient="records")


def load_json(path: Union[Path, str]) -> Dict:
    path = Path(path)
    with path.open() as f:
        data = json.load(f)
    return data


def save_config(data, path: Union[Path, str]) -> None:
    if type(data) != dict:
        data = vars(data)
    save_json(
        {k: v if type(v) in [int, float, bool, None] else str(v) for k, v in data.items()},
        path,
    )
