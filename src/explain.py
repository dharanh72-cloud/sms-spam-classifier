"""Model-agnostic explanation: which words push a message towards / away from SPAM?"""
import numpy as np

from .inference import load


def explain(message: str, top: int = 5):
    """Occlusion: drop each word and measure the change in spam log-odds (works even when p is saturated).
    Returns [(word, delta)], largest effect first; positive = pushes towards SPAM."""
    model, _ = load()
    words = message.split()
    if len(words) < 2:
        return []
    variants = [message] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    p = np.clip(model.predict_proba(variants)[:, 1], 1e-4, 1 - 1e-4)
    probs = np.log(p / (1 - p))
    deltas = [(w, float(probs[0] - p)) for w, p in zip(words, probs[1:])]
    return sorted(deltas, key=lambda x: -abs(x[1]))[:top]
