"""Obfuscation augmentation: spammers write 'fr33', 'w.i.n', 'pr ize' to dodge filters."""
import random

_LEET = str.maketrans({"a": "@", "e": "3", "i": "1", "o": "0", "s": "5"})


def obfuscate(text: str, rng: random.Random) -> str:
    out = []
    for w in text.split():
        r = rng.random()
        if len(w) > 3 and r < 0.30:
            w = w.translate(_LEET)                       # leetspeak
        elif len(w) > 3 and r < 0.45:
            w = ".".join(w)                              # w.i.n
        elif len(w) > 3 and r < 0.55:
            w = w[: len(w) // 2] + " " + w[len(w) // 2:]  # pr ize
        out.append(w)
    return " ".join(out)


def augment_spam(X, y, seed: int = 0):
    """Return X, y plus one obfuscated copy of every spam message (train data only!)."""
    rng = random.Random(seed)
    X, y = list(X), list(y)
    extra = [obfuscate(t, rng) for t, l in zip(X, y) if l == 1]
    return X + extra, y + [1] * len(extra)
