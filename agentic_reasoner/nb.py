import json
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

from .text import featurize


class MultinomialNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_counts: Dict[str, int] = {}
        self.feature_counts: Dict[str, Counter] = {}
        self.vocab: set[str] = set()
        self.class_priors: Dict[str, float] = {}
        self.feature_log_probs: Dict[str, Dict[str, float]] = {}
        self.fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> None:
        assert len(texts) == len(labels)
        class_counts = Counter(labels)
        feature_counts: Dict[str, Counter] = defaultdict(Counter)
        vocab = set()

        for text, y in zip(texts, labels):
            feats = featurize(text)
            vocab.update(feats)
            feature_counts[y].update(feats)

        self.class_counts = dict(class_counts)
        self.feature_counts = {c: Counter(cnt) for c, cnt in feature_counts.items()}
        self.vocab = vocab

        # Compute priors and likelihoods
        total_docs = sum(class_counts.values())
        self.class_priors = {c: math.log(class_counts[c] / total_docs) for c in class_counts}

        self.feature_log_probs = {}
        V = len(vocab) or 1
        for c, cnt in self.feature_counts.items():
            total = sum(cnt.values())
            denom = total + self.alpha * V
            self.feature_log_probs[c] = {}
            for f in vocab:
                num = cnt.get(f, 0) + self.alpha
                self.feature_log_probs[c][f] = math.log(num / denom)

        self.fitted = True

    def predict_proba(self, text: str) -> Dict[str, float]:
        assert self.fitted
        feats = featurize(text)
        class_scores: Dict[str, float] = {}
        for c in self.class_priors:
            score = self.class_priors[c]
            for f in feats:
                if f in self.feature_log_probs[c]:
                    score += self.feature_log_probs[c][f]
            class_scores[c] = score
        # Convert log-scores to probabilities (softmax)
        m = max(class_scores.values())
        exps = {c: math.exp(v - m) for c, v in class_scores.items()}
        Z = sum(exps.values())
        return {c: v / Z for c, v in exps.items()}

    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        probs = self.predict_proba(text)
        best = max(probs.items(), key=lambda kv: kv[1])[0]
        return best, probs

    def top_features_for(self, text: str, cls: str, k: int = 10) -> List[Tuple[str, float]]:
        """Return tokens from the given text that most increase the score for class 'cls'."""
        assert self.fitted
        feats = featurize(text)
        contribs = []
        for f in feats:
            lp = self.feature_log_probs.get(cls, {}).get(f, None)
            if lp is not None:
                contribs.append((f, lp))
        contribs.sort(key=lambda x: x[1], reverse=True)
        return contribs[:k]

    def save(self, path: str) -> None:
        obj: Dict[str, Any] = {
            "alpha": self.alpha,
            "class_counts": self.class_counts,
            "feature_counts": {c: dict(cnt) for c, cnt in self.feature_counts.items()},
            "vocab": list(self.vocab),
            "class_priors": self.class_priors,
            "feature_log_probs": self.feature_log_probs,
            "fitted": self.fitted,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    @classmethod
    def load(cls, path: str) -> "MultinomialNB":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        m = cls(alpha=obj.get("alpha", 1.0))
        m.class_counts = obj["class_counts"]
        m.feature_counts = {c: Counter(d) for c, d in obj["feature_counts"].items()}
        m.vocab = set(obj["vocab"])
        m.class_priors = {k: float(v) for k, v in obj["class_priors"].items()}
        m.feature_log_probs = {
            c: {k: float(v) for k, v in d.items()} for c, d in obj["feature_log_probs"].items()
        }
        m.fitted = obj["fitted"]
        return m