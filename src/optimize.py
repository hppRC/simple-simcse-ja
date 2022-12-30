from datetime import datetime
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
from src.dataset import SupSimCSEDataset, UnsupSimCSEDataset
from src.models import SimCSEModel
from src.sts import STSEvaluation
from src.train import Args, Experiment, main

if __name__ == "__main__":
    args = Args.from_dict()
    main(args)
