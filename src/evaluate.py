import torch.nn as nn
from sentence_transformers import SentenceTransformer, models
from src.sts import STSEvaluation
from transformers import AutoModel, BertModel

# MODEL_PATH = "cl-nagoya/sup-simcse-ja-large"
# MODEL_PATH = "cl-nagoya/sup-simcse-ja-base"
# MODEL_PATH = "MU-Kindai/Japanese-SimCSE-BERT-large-sup"
# MODEL_PATH = "colorfulscoop/sbert-base-ja"
# MODEL_PATH = "pkshatech/GLuCoSE-base-ja"
# MODEL_PATH = "oshizo/sbert-jsnli-luke-japanese-base-lite"
MODEL_PATH = "intfloat/multilingual-e5-large"


sts = STSEvaluation(sts_dir="./datasets/sts")


def evaluate():
    model = SentenceTransformer(MODEL_PATH).eval().cuda()
    print(sts.dev(encode=model.encode))
    print(sts(encode=model.encode))


def evaluate_jcse():
    backbone = models.Transformer(MODEL_PATH)
    pretrained_model: BertModel = AutoModel.from_pretrained(MODEL_PATH)
    hidden_size = pretrained_model.config.hidden_size

    # load weights of Transformer layers
    backbone.auto_model.load_state_dict(pretrained_model.state_dict())
    pooling = models.Pooling(
        word_embedding_dimension=hidden_size,
        pooling_mode="cls",
    )

    if "unsup" in MODEL_PATH:
        model = SentenceTransformer(modules=[backbone, pooling]).eval().cuda()

    else:
        # load weights of extra MLP layer
        # unsupervised models do not use this, so we need to create a new one only for supervised models
        mlp = models.Dense(
            in_features=hidden_size,
            out_features=hidden_size,
            activation_function=nn.Tanh(),
        )
        mlp_state_dict = {
            key.replace("dense.", "linear."): param
            for key, param in pretrained_model.pooler.state_dict().items()
        }
        mlp.load_state_dict(mlp_state_dict)
        model = SentenceTransformer(modules=[backbone, pooling, mlp]).eval().cuda()

    print(sts.dev(encode=model.encode))
    print(sts(encode=model.encode))


def evaluate_vanilla():
    backbone = models.Transformer(MODEL_PATH)
    pooling = models.Pooling(
        word_embedding_dimension=backbone.auto_model.config.hidden_size,
        pooling_mode="cls",
    )
    model = SentenceTransformer(modules=[backbone, pooling]).eval().cuda()
    print(sts.dev(encode=model.encode))
    print(sts(encode=model.encode))


def evaluate_openai():
    import os
    import openai
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from more_itertools import chunked
    from openai.openai_object import OpenAIObject

    openai.api_key = os.environ["OPENAI_API_KEY"]

    def encode_openai(batch: list[str]):
        res: OpenAIObject = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=batch,
        )
        return [d.embedding for d in res.data]

    def encode(sentences: list[str], batch_size: int = 128):
        embs = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            batches = chunked(list(sentences), batch_size)
            for emb in executor.map(encode_openai, batches):
                embs += emb
        embs = np.array(embs)
        return embs

    print(sts.dev(encode=encode))
    print(sts(encode=encode))


def evaluate_e5():
    model = SentenceTransformer(MODEL_PATH).eval().cuda()

    def encode(sentences: list[str]):
        sentences = [f"query: {s}" for s in sentences]
        return model.encode(sentences)

    print(sts.dev(encode=encode))
    print(sts(encode=encode))


if __name__ == "__main__":
    # evaluate()
    # evaluate_vanilla()
    # evaluate_openai()
    evaluate_e5()
