"""Load the saved model and classify messages into SPAM / HAM / REVIEW."""
import json

import joblib

from .data import ROOT

_cache = {}


def load():
    if "model" not in _cache:
        model_path = ROOT / "models" / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError("models/model.joblib not found - run `python train.py` first.")
        _cache["model"] = joblib.load(model_path)
        _cache["cfg"] = json.loads((ROOT / "models" / "config.json").read_text())
    return _cache["model"], _cache["cfg"]


def classify(messages):
    """Return a list of dicts: {message, spam_probability, decision}.

    decision is SPAM (p >= threshold), HAM (p < review_low) or REVIEW (in between:
    the model is unsure, so route to a human or a rule-based check)."""
    model, cfg = load()
    probs = model.predict_proba(list(messages))[:, 1]
    out = []
    for msg, p in zip(messages, probs):
        if p >= cfg["threshold"]:
            label = "SPAM"
        elif p < cfg["review_low"]:
            label = "HAM"
        else:
            label = "REVIEW"
        out.append({"message": msg, "spam_probability": float(p), "decision": label})
    return out
