import pytest

from src.data import load_data
from src.features import HandcraftedFeatures, build_pipeline
from src.preprocessing import clean_text


def test_dataset_deduplicated():
    df = load_data()
    assert not df.message.duplicated().any()
    assert set(df.label) == {"ham", "spam"}


def test_clean_text_removes_noise():
    assert clean_text("FREE entry!!! http://x.com 12345 to win") == "free entry win"


def test_handcrafted_features_shape_and_flags():
    hf = HandcraftedFeatures().fit(["hello", "WIN £500 now!!! call 09061701461"])
    X = hf.transform(["WIN £500 now!!! call 09061701461"]).toarray()[0]
    assert X.shape == (8,)
    assert X[6] > 0 and X[7] > 0  # currency + long-number flags


def test_pipeline_fits_and_predicts_tiny_data():
    X = ["free prize win now", "call me later", "win cash free txt", "see you at lunch"] * 5
    y = [1, 0, 1, 0] * 5
    pipe = build_pipeline("word+char+hand", "LinearSVM").fit(X, y)
    assert list(pipe.predict(["free cash prize"])) == [1]


def test_saved_model_end_to_end():
    from src.inference import classify
    try:
        res = classify(["FREE entry! Text WIN to 80086 now to claim your prize", "are we still on for lunch?"])
    except FileNotFoundError:
        pytest.skip("run train.py first")
    assert res[0]["decision"] == "SPAM"
    assert res[1]["decision"] == "HAM"


def test_html_entities_are_unescaped():
    from src.data import normalize
    assert normalize("call  &lt;#&gt; &amp; win") == "call <#> & win"


def test_near_duplicates_share_a_group():
    from src.data import near_duplicate_groups
    g = near_duplicate_groups([
        "WIN a FREE prize call 09061701461 now claim",
        "WIN a FREE prize call 09061701999 now claim",
        "are we still meeting for lunch tomorrow",
    ])
    assert g[0] == g[1] != g[2]


def test_obfuscation_changes_text_but_augment_only_adds_spam():
    import random

    from src.augment import augment_spam, obfuscate
    assert obfuscate("free prize message now please", random.Random(3)) != "free prize message now please"
    X, y = augment_spam(["free prize now", "see you soon"], [1, 0])
    assert len(X) == 3 and y == [1, 0, 1]


def test_explain_returns_words():
    from src.explain import explain
    try:
        top = explain("FREE entry! Text WIN to 80086 now to claim your prize")
    except FileNotFoundError:
        pytest.skip("run train.py first")
    assert 0 < len(top) <= 5 and all(isinstance(w, str) for w, _ in top)


def test_api_endpoints():
    from fastapi.testclient import TestClient

    from api import app
    c = TestClient(app)
    try:
        r = c.post("/classify", json={"text": "FREE entry! Text WIN to 80086 now", "explain": True})
    except FileNotFoundError:
        pytest.skip("run train.py first")
    assert r.status_code == 200 and r.json()["decision"] in {"SPAM", "REVIEW"} and "top_words" in r.json()
    assert c.post("/classify", json={"text": ""}).status_code == 422


def test_decision_bands(monkeypatch):
    """SPAM >= threshold, HAM < review_low, REVIEW in between - independent of any saved model."""
    import numpy as np

    from src import inference

    class Stub:
        def predict_proba(self, msgs):
            p = np.array([0.9, 0.1, 0.01, 0.217, 0.024])[: len(msgs)]
            return np.c_[1 - p, p]

    monkeypatch.setattr(inference, "load", lambda: (Stub(), {"threshold": 0.217, "review_low": 0.024}))
    out = inference.classify(["a", "b", "c", "d", "e"])
    assert [r["decision"] for r in out] == ["SPAM", "REVIEW", "HAM", "SPAM", "REVIEW"]
