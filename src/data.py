import html
import re
from pathlib import Path

import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "SMSSpamCollection.tsv"
SEED = 42


def normalize(text: str) -> str:
    """Un-escape HTML entities (the dataset contains '&lt;#&gt;' / '&amp;' leftovers) and squeeze whitespace."""
    return re.sub(r"\s+", " ", html.unescape(str(text))).strip()


def near_duplicate_groups(texts, threshold: float = 0.9):
    """Group templated / near-identical messages (char n-gram cosine >= threshold, digits masked).

    Spam is heavily templated, so a message and its near-copy must never be split across
    train and test - use these ids as `groups` in StratifiedGroupKFold."""
    masked = [re.sub(r"\d", "0", t.lower()) for t in texts]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit_transform(masked)
    _, labels = connected_components((vec @ vec.T) >= threshold, directed=False)
    return labels


def load_data(deduplicate: bool = True) -> pd.DataFrame:
    """Load the SMS Spam Collection, normalise text, drop exact duplicates and add a `group` id."""
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    df["message"] = df["message"].map(normalize)
    df["label_num"] = (df["label"] == "spam").astype(int)
    if deduplicate:
        df = df.drop_duplicates(subset="message").reset_index(drop=True)
    df["group"] = near_duplicate_groups(df["message"].tolist())
    return df
