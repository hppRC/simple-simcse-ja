import re
import unicodedata

from konoha import SentenceTokenizer

tokenizer = SentenceTokenizer(
    patterns=[
        re.compile(r"（.*?）"),
        re.compile(r"\(.*?\)"),
        re.compile(r"「.*?」"),
        re.compile(r"『.*?』"),
    ],
)


def preprocess_text(text: str) -> list[str]:
    text: str = unicodedata.normalize("NFKC", text).strip()
    text: str = "".join(c for c in text if c.isprintable())
    sentences: list[str] = [s.strip() for s in tokenizer.tokenize(text)]
    sentences: list[str] = [s for s in sentences if len(s) > 1]
    return sentences
