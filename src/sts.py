from pathlib import Path
from typing import Callable, Dict, List, Union

import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances
from torch import Tensor
from tqdm import tqdm

# TODO: さまざまな距離関数による評価


class STSEvaluatorBase:
    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        scores: List[float],
    ):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        assert len(self.sentences1) == len(self.sentences2) == len(self.scores)

    def __call__(self, encode: Callable[[List[str]], Tensor]) -> float:
        embeddings1 = encode(self.sentences1)
        embeddings2 = encode(self.sentences2)
        # you can use any similarity function you want ↓
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        spearman = float(spearmanr(self.scores, cosine_scores)[0]) * 100

        return spearman


class JSICKEvaluator(STSEvaluatorBase):
    # Title: Compositional Evaluation on Japanese Textual Entailment and Similarity
    # URL: https://arxiv.org/abs/2208.04826
    # GitHub: https://github.com/verypluming/JSICK

    def __init__(self, sts_dir: Path):
        df = pd.read_table(sts_dir / "jsick/jsick/jsick.tsv", sep="\t")
        df = df[df["data"] == "test"]
        sentences1 = df["sentence_A_Ja"].values
        sentences2 = df["sentence_B_Ja"].values
        scores = df["relatedness_score_Ja"].values

        super().__init__(sentences1, sentences2, scores)


class JSICKTrainEvaluator(STSEvaluatorBase):
    def __init__(self, sts_dir: Path):
        df = pd.read_table(sts_dir / "jsick/jsick/jsick.tsv", sep="\t")
        df = df[df["data"] == "train"]
        sentences1 = df["sentence_A_Ja"].values
        sentences2 = df["sentence_B_Ja"].values
        scores = df["relatedness_score_Ja"].values

        super().__init__(sentences1, sentences2, scores)


class JSTSValidEvaluator(STSEvaluatorBase):
    # Title: JGLUE: Japanese General Language Understanding Evaluation
    # URL: https://aclanthology.org/2022.lrec-1.317/
    # Title (Japanese): JGLUE: 日本語言語理解ベンチマーク
    # URL: https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E8-4.pdf
    # GitHub: https://github.com/yahoojapan/JGLUE

    def __init__(self, sts_dir: Path):
        df = pd.read_json(sts_dir / "jsts/valid-v1.1.json", lines=True)
        sentences1 = df["sentence1"].values
        sentences2 = df["sentence2"].values
        scores = df["label"].values

        super().__init__(sentences1, sentences2, scores)


class JSTSTrainEvaluator(STSEvaluatorBase):
    def __init__(self, sts_dir: Path):
        df = pd.read_json(sts_dir / "jsts/train-v1.1.json", lines=True)
        sentences1 = df["sentence1"].values
        sentences2 = df["sentence2"].values
        scores = df["label"].values

        super().__init__(sentences1, sentences2, scores)


class STSEvaluation:
    def __init__(self, sts_dir: Union[str, Path]):
        sts_dir = Path(sts_dir)
        self.sts_evaluators = {
            "jsick": JSICKEvaluator(sts_dir=sts_dir),
            "jsts-val": JSTSValidEvaluator(sts_dir=sts_dir),
            "jsts-train": JSTSTrainEvaluator(sts_dir=sts_dir),
        }
        self.dev_evaluator = JSICKTrainEvaluator(sts_dir=sts_dir)

    @torch.inference_mode()
    def __call__(
        self,
        encode: Callable[[List[str]], Tensor],
        progress_bar: bool = True,
    ) -> Dict[str, float]:

        if progress_bar:
            iterator = tqdm(
                list(self.sts_evaluators.items()),
                dynamic_ncols=True,
                leave=False,
            )
        else:
            iterator = list(self.sts_evaluators.items())

        results = {}
        for name, evaluator in iterator:
            results[name] = evaluator(encode=encode)

        results["avg"] = sum(results.values()) / len(results)
        return results

    @torch.inference_mode()
    def dev(
        self,
        encode: Callable[[List[str]], Tensor],
    ) -> float:
        return self.dev_evaluator(encode=encode)
