from typing import Literal

import torch.nn as nn
from torch import FloatTensor, LongTensor
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

MLPType = Literal["simcse", "diffcse"]


class SimMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.dense(x)
        x = self.activation(x)
        return x


class DiffMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        intermidiate_size = hidden_size * 2

        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermidiate_size, bias=False),
            nn.BatchNorm1d(intermidiate_size),
            nn.ReLU(inplace=True),
            nn.Linear(intermidiate_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size, affine=False),
        )

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.net(x)


class SimCSEModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        mlp_type: MLPType = "simcse",
        mlp_only_train: bool = False,
    ):
        super().__init__()
        self.mlp_only_train = mlp_only_train
        self.backbone: PreTrainedModel = AutoModel.from_pretrained(model_name)
        self.hidden_size: int = self.backbone.config.hidden_size

        match mlp_type:
            case "simcse":
                self.mlp = SimMLP(self.hidden_size)
            case "diffcse":
                self.mlp = DiffMLP(self.hidden_size)
            case _:
                raise ValueError(f"Unknown mlp_type: {mlp_type}")

    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: LongTensor = None,
        token_type_ids: LongTensor = None,
    ) -> FloatTensor:
        outputs: BaseModelOutput = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        emb: FloatTensor = outputs.last_hidden_state[:, 0]

        if self.training or not self.mlp_only_train:
            emb = self.mlp(emb)
        return emb
