import re
from typing import List, Iterable, Tuple

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

STOPWORDS = set(
    """
    a an the and or if then else when while for to of in on at by with from into over under about after before
    is are was were be been being this that these those it its as not no yes do does did doing done can could
    would should may might must will shall you your yours me my mine we our ours they their theirs he she his her
    him hers who whom which what where why how there here out up down left right forward backward back again
    """.split()
)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = _WORD_RE.findall(text)
    return tokens


def filter_tokens(tokens: Iterable[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit()]


def ngrams(tokens: List[str], n: int = 1) -> List[str]:
    if n <= 1:
        return list(tokens)
    return ["_".join(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1))]


def featurize(text: str, use_bigrams: bool = True) -> List[str]:
    toks = filter_tokens(tokenize(text))
    feats = toks
    if use_bigrams:
        feats = feats + ngrams(toks, 2)
    return feats


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()