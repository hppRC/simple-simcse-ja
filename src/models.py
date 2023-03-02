import torch.nn as nn
from torch import FloatTensor, LongTensor
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput


class SimCSEModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        mlp_only_train: bool = False,
    ):
        super().__init__()
        self.mlp_only_train = mlp_only_train
        self.backbone: PreTrainedModel = AutoModel.from_pretrained(
            model_name,
            torch_dtype="auto",
        )

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
            token_type_ids=token_type_ids,
        )
        emb: FloatTensor = outputs.last_hidden_state[:, 0]

        if self.training or not self.mlp_only_train:
            emb = self.dense(emb)
            emb = self.activation(emb)
        return emb
