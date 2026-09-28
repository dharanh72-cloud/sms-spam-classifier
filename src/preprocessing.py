"""Text cleaning used ONLY for the 'baseline' model that mimics the original project.

The improved models work on the raw message text, because digits, punctuation,
capitalisation and currency symbols are strong spam signals that heavy cleaning removes.
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_NON_LETTER = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Lowercase, drop URLs / digits / punctuation, remove English stopwords."""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _NON_LETTER.sub("", text)
    return " ".join(t for t in text.split() if t not in ENGLISH_STOP_WORDS)


def clean_series(texts):
    """Vectorised helper so it can be used inside a scikit-learn FunctionTransformer."""
    return [clean_text(t) for t in texts]
