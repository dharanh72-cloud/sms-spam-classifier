"""Hand-crafted features + pipeline builders."""
import re

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer  # noqa: F401  (kept for easy extension)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from .preprocessing import clean_series

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_LONG_NUMBER = re.compile(r"\d{5,}")          # phone numbers, short codes, prize codes
_CURRENCY = re.compile(r"[£$€]|\b(?:gbp|usd|eur|rs)\b", re.I)

HAND_FEATURE_NAMES = [
    "log_length", "log_words", "digit_ratio", "upper_ratio",
    "log_exclaims", "has_url", "has_currency", "has_long_number",
]


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Length, digit/uppercase ratios, '!' count, URL / currency / long-number flags."""

    def fit(self, X, y=None):
        raw = self._raw(X)
        self.scale_ = np.maximum(np.abs(raw).max(axis=0), 1e-9)
        return self

    def transform(self, X):
        return sparse.csr_matrix(self._raw(X) / self.scale_)

    @staticmethod
    def _raw(X):
        rows = []
        for t in X:
            t = str(t)
            n = max(len(t), 1)
            rows.append([
                np.log1p(len(t)),
                np.log1p(len(t.split())),
                sum(c.isdigit() for c in t) / n,
                sum(c.isupper() for c in t) / n,
                np.log1p(t.count("!")),
                float(bool(_URL.search(t))),
                float(bool(_CURRENCY.search(t))),
                float(bool(_LONG_NUMBER.search(t))),
            ])
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(HAND_FEATURE_NAMES)


def _word():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)


def _char():
    # char_wb n-grams are robust to obfuscation such as "fr33", "w1n", "txt" vs "text"
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                           lowercase=False, min_df=2)


def build_features(kind: str):
    if kind == "baseline_clean":      # mimics the original project
        return Pipeline([
            ("clean", FunctionTransformer(clean_series)),
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ])
    if kind == "word":
        return _word()
    if kind == "char":
        return _char()
    if kind == "word+char":
        return make_union(_word(), _char())
    if kind == "word+char+hand":
        return make_union(_word(), _char(), HandcraftedFeatures())
    raise ValueError(kind)


def build_model(name: str, C: float = 1.0):
    if name == "LinearSVM":
        return LinearSVC(C=C, class_weight="balanced")
    if name == "LogisticRegression":
        return LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    if name == "ComplementNB":
        return ComplementNB(alpha=1.0 / max(C, 1e-6) if C else 1.0)
    raise ValueError(name)


def build_pipeline(kind: str, model: str, C: float = 1.0) -> Pipeline:
    return Pipeline([("features", build_features(kind)), ("model", build_model(model, C))])


FEATURE_SETS = ["baseline_clean", "word", "char", "word+char", "word+char+hand"]
MODELS = ["ComplementNB", "LogisticRegression", "LinearSVM"]
