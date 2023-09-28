import torch.nn as nn
from peft import LoraModel
from torch import FloatTensor, LongTensor
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput


class Pooler:
    def __init__(
        self,
        pooling: str = "cls",
    ):
        self.pooling = pooling

    def cls_pooling(
        self,
        last_hidden_state: FloatTensor,
        attention_mask: FloatTensor,
    ) -> FloatTensor:
        emb: FloatTensor = last_hidden_state[:, 0]
        return emb

    def mean_pooling(
        self,
        last_hidden_state: FloatTensor,
        attention_mask: FloatTensor,
    ) -> FloatTensor:
        sent_len = attention_mask.sum(dim=1, keepdim=True)
        emb = last_hidden_state * attention_mask.unsqueeze(-1)
        emb = emb.sum(dim=1) / sent_len
        return emb

    def __call__(self, *args, **kwargs) -> FloatTensor:
        if self.pooling == "cls":
            return self.cls_pooling(*args, **kwargs)
        elif self.pooling == "mean":
            return self.mean_pooling(*args, **kwargs)
        else:
            raise ValueError(f"pooling must be 'cls' or 'mean', but got {self.pooling}")


class SimCSEModel(nn.Module):
    backbone: PreTrainedModel | LoraModel

    def __init__(
        self,
        model_name: str = None,
        backbone: PreTrainedModel = None,
        mlp_only_train: bool = True,
        gradient_checkpointing: bool = False,
        pooling: str = "cls",
    ):
        super().__init__()
        self.mlp_only_train = mlp_only_train

        if model_name is None and backbone is None:
            raise ValueError("model_name and backbone cannot be both None")

        if backbone is None:
            self.backbone: PreTrainedModel = AutoModel.from_pretrained(
                model_name,
                torch_dtype="auto",
            )
        else:
            self.backbone = backbone

        self.pooling = Pooler(pooling)

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        hidden_size: int = self.backbone.config.hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: LongTensor = None,
        token_type_ids: LongTensor = None,
    ) -> FloatTensor:
        outputs: BaseModelOutput = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        emb = self.pooling(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )

        if self.training or not self.mlp_only_train:
            emb = self.dense(emb)
            emb = self.activation(emb)
        return emb
