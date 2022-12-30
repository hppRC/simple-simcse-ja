from typing import Literal

import torch.nn as nn
from torch import FloatTensor, LongTensor
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

MLPType = Literal[
    "bn",
    "simple",
    "simcse",
    "simcse-pre-bn",
    "simcse-mid-bn",
    "simcse-post-bn",
    "simcse-taper",
    "diffcse",
    "simsiam",
]


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bn = nn.BatchNorm1d(hidden_size, affine=False)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.dense(x)
        x = self.bn(x)
        return x


class SimCSEPreBNMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_size, affine=False)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.bn(x)
        x = self.dense(x)
        x = self.activation(x)
        return x


class SimCSEMidBNMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size, affine=False)
        self.activation = nn.Tanh()

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.dense(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SimCSEPostBNMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.bn = nn.BatchNorm1d(hidden_size, affine=False)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.dense(x)
        x = self.activation(x)
        x = self.bn(x)
        return x


class SimCSETaperMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size // 4)
        self.activation = nn.Tanh()
        self.bn = nn.BatchNorm1d(hidden_size // 4, affine=False)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.dense(x)
        x = self.activation(x)
        x = self.bn(x)
        return x


class SimCSEMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.dense(x)
        x = self.activation(x)
        return x


class DiffCSEMLP(nn.Module):
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


class SimSiamMLP(nn.Module):
    # https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py

    def __init__(self, hidden_size: int):
        super().__init__()
        intermidiate_size = hidden_size * 2

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),  # second layer
            nn.Linear(hidden_size, intermidiate_size, bias=False),
            nn.BatchNorm1d(intermidiate_size, affine=False),
        )

        self.predictor = nn.Sequential(
            nn.Linear(intermidiate_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(hidden_size, hidden_size),
        )  # output layer

    def forward(self, x: FloatTensor) -> FloatTensor:
        z = self.fc(x)
        p = self.predictor(z)
        # return p, z.detach()
        return p


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
            case "bn":
                self.mlp = nn.BatchNorm1d(self.hidden_size, affine=False)
            case "simple":
                self.mlp = SimpleMLP(self.hidden_size)
            case "simcse":
                self.mlp = SimCSEMLP(self.hidden_size)
            case "simcse-pre-bn":
                self.mlp = SimCSEPreBNMLP(self.hidden_size)
            case "simcse-mid-bn":
                self.mlp = SimCSEMidBNMLP(self.hidden_size)
            case "simcse-post-bn":
                self.mlp = SimCSEPostBNMLP(self.hidden_size)
            case "simcse-taper":
                self.mlp = SimCSETaperMLP(self.hidden_size)
            case "diffcse":
                self.mlp = DiffCSEMLP(self.hidden_size)
            case "simsiam":
                self.mlp = SimSiamMLP(self.hidden_size)
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
