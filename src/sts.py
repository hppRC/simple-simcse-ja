from pathlib import Path
from typing import Callable

import torch
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances
from src import utils
from torch import FloatTensor
from tqdm import tqdm

# TODO: さまざまな距離関数による評価


class STSEvaluatorBase:
    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        scores: list[float],
        do_jumanpp_preprocess: bool = False,
    ):
        self.sentences1 = sentences1
        self.sentences2 = sentences2

        if do_jumanpp_preprocess:
            self.sentences1 = [utils.jumanpp_wakati(sent) for sent in self.sentences1]
            self.sentences2 = [utils.jumanpp_wakati(sent) for sent in self.sentences2]

        self.scores = scores
        assert len(self.sentences1) == len(self.sentences2) == len(self.scores)

    def __call__(self, encode: Callable[[list[str]], FloatTensor]) -> float:
        embeddings1 = encode(self.sentences1)
        embeddings2 = encode(self.sentences2)

        # you can use any similarity function you want ↓
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        spearman, _ = spearmanr(self.scores, cosine_scores)
        spearman = float(spearman) * 100
        return spearman


class JSICKEvaluator(STSEvaluatorBase):
    # Title: Compositional Evaluation on Japanese Textual Entailment and Similarity
    # URL: https://arxiv.org/abs/2208.04826
    # GitHub: https://github.com/verypluming/JSICK

    def __init__(self, sts_dir: Path, *args, **kwargs):
        df = utils.load_jsonl(sts_dir / "jsick/test.jsonl")
        super().__init__(df["sent0"], df["sent1"], df["score"], *args, **kwargs)


class JSICKTrainEvaluator(STSEvaluatorBase):
    def __init__(self, sts_dir: Path, *args, **kwargs):
        df = utils.load_jsonl(sts_dir / "jsick/train.jsonl")
        super().__init__(df["sent0"], df["sent1"], df["score"], *args, **kwargs)


class JSTSValidEvaluator(STSEvaluatorBase):
    # Title: JGLUE: Japanese General Language Understanding Evaluation
    # URL: https://aclanthology.org/2022.lrec-1.317/
    # Title (Japanese): JGLUE: 日本語言語理解ベンチマーク
    # URL: https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E8-4.pdf
    # GitHub: https://github.com/yahoojapan/JGLUE

    def __init__(self, sts_dir: Path, *args, **kwargs):
        df = utils.load_jsonl(sts_dir / "jsts/val.jsonl")
        super().__init__(df["sent0"], df["sent1"], df["score"], *args, **kwargs)


class JSTSTrainEvaluator(STSEvaluatorBase):
    def __init__(self, sts_dir: Path, *args, **kwargs):
        df = utils.load_jsonl(sts_dir / "jsts/train.jsonl")
        super().__init__(df["sent0"], df["sent1"], df["score"], *args, **kwargs)


class STSEvaluation:
    def __init__(
        self,
        sts_dir: str | Path,
        do_jumanpp_preprocess: bool = False,
    ):
        sts_dir = Path(sts_dir)
        self.sts_evaluators = {
            "jsick": JSICKEvaluator(
                sts_dir=sts_dir,
                do_jumanpp_preprocess=do_jumanpp_preprocess,
            ),
            "jsts-val": JSTSValidEvaluator(
                sts_dir=sts_dir,
                do_jumanpp_preprocess=do_jumanpp_preprocess,
            ),
            "jsts-train": JSTSTrainEvaluator(
                sts_dir=sts_dir,
                do_jumanpp_preprocess=do_jumanpp_preprocess,
            ),
        }
        self.dev_evaluator = JSICKTrainEvaluator(
            sts_dir=sts_dir,
            do_jumanpp_preprocess=do_jumanpp_preprocess,
        )

    @torch.inference_mode()
    def __call__(
        self,
        encode: Callable[[list[str]], FloatTensor],
        progress_bar: bool = True,
    ) -> dict[str, float]:
        if progress_bar:
            iterator = tqdm(list(self.sts_evaluators.items()), dynamic_ncols=True, leave=False)
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
        encode: Callable[[list[str]], FloatTensor],
    ) -> float:
        return self.dev_evaluator(encode=encode)
