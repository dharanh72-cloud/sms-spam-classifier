"""Train, select, tune, calibrate and evaluate the spam filter.

Run:  python train.py
Design (what fixes the original project's weaknesses):
  1. Exact duplicates removed and near-duplicate templates GROUPED, so they never straddle train/test.
  2. ~80/20 stratified group split; the test set is touched ONCE, at the very end.
  3. Model selection by repeated stratified cross-validation on the training set only.
  4. Hyper-parameter (C) tuning with GridSearchCV on the training set only.
  5. Probability calibration + decision threshold chosen on out-of-fold predictions to hit a
     target spam PRECISION (a false positive blocks a real message), then maximise recall.
  6. Low-confidence "review" band instead of forcing a hard decision.
  7. Bootstrap 95% confidence intervals on the test metrics.
  8. Spam is augmented with obfuscated copies (fr33, w.i.n ...) in TRAINING data only; robustness is
     measured on an obfuscated version of the test spam.
"""
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate

from src.augment import augment_spam, obfuscate
from src.data import ROOT, SEED, load_data
from src.features import FEATURE_SETS, MODELS, build_pipeline

warnings.filterwarnings("ignore")
OUT, MODELS_DIR = ROOT / "outputs", ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

TARGET_PRECISION = 0.98      # business rule: at most ~2% of messages flagged as spam are actually ham
REVIEW_RATE = 0.05           # aim to send ~5% of messages (the most uncertain ones) to human review
C_GRID = [0.1, 0.3, 1, 3, 10, 30]


def spam_metrics(y, pred):
    return dict(precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred))


def bootstrap_ci(y, pred, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    stats = {"precision": [], "recall": [], "f1": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        m = spam_metrics(y[i], pred[i])
        for k in stats:
            stats[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}


def main():
    df = load_data(deduplicate=True)
    X, y = df["message"], df["label_num"]
    g = df["group"]
    tr_i, te_i = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g))
    X_tr, X_te, y_tr, y_te, g_tr = X.iloc[tr_i], X.iloc[te_i], y.iloc[tr_i], y.iloc[te_i], g.iloc[tr_i]
    print(f"Messages after de-duplication: {len(df)}  (spam {y.mean():.1%})")
    print(f"Train {len(X_tr)} | Test {len(X_te)} (test spam: {int(y_te.sum())})")

    # ---- 1. Model selection by repeated stratified CV (train set only) -------------------------
    cv = [s for r in range(2) for s in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(X_tr, y_tr, g_tr)]
    rows = []
    for fs in FEATURE_SETS:
        for m in MODELS:
            if m == "ComplementNB" and fs == "word+char+hand":
                continue
            res = cross_validate(build_pipeline(fs, m), X_tr, y_tr, cv=cv, n_jobs=-1,
                                 scoring={"precision": "precision", "recall": "recall", "f1": "f1"})
            rows.append(dict(features=fs, model=m,
                             precision=res["test_precision"].mean(), recall=res["test_recall"].mean(),
                             f1=res["test_f1"].mean(), f1_std=res["test_f1"].std()))
            print(f"  CV {fs:15} {m:19} F1={rows[-1]['f1']:.4f} +/- {rows[-1]['f1_std']:.4f}")
    cv_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    cv_df.to_csv(OUT / "cv_results.csv", index=False)
    best = cv_df.iloc[0]
    print(f"\nBest by CV: {best.features} + {best.model} (F1={best.f1:.4f})")

    # ---- 2. Tune C on train only ----------------------------------------------------------------
    grid = GridSearchCV(build_pipeline(best.features, best.model), {"model__C": C_GRID}
                        if best.model != "ComplementNB" else {"model__alpha": [0.03, 0.1, 0.3, 1.0]},
                        scoring="f1", cv=list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X_tr, y_tr, g_tr)), n_jobs=-1)
    grid.fit(X_tr, y_tr)
    print("Tuned params:", grid.best_params_, f"(CV F1={grid.best_score_:.4f})")
    tuned = grid.best_estimator_

    # ---- 3. Calibrate + choose threshold on out-of-fold probabilities ---------------------------
    def fit_calibrated(Xa, ya, augment=True):
        if augment:
            Xa, ya = augment_spam(Xa, ya, seed=SEED)
        return CalibratedClassifierCV(tuned, method="sigmoid", cv=5).fit(Xa, ya)

    oof = np.zeros(len(X_tr))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X_tr, y_tr, g_tr):
        oof[b] = fit_calibrated(X_tr.iloc[a], y_tr.iloc[a]).predict_proba(X_tr.iloc[b])[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, oof)
    ok = np.where(prec[:-1] >= TARGET_PRECISION)[0]
    if len(ok):
        threshold = float(thr[ok[np.argmax(rec[:-1][ok])]])
    else:  # fall back to best F1 if the precision target is unreachable
        f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
        threshold = float(thr[np.argmax(f1s)])
    below = np.sort(oof[oof < threshold])
    k = max(int(REVIEW_RATE * len(oof)), 1)
    review_low = float(below[-k]) if len(below) >= k else float(below[0])
    print(f"Threshold (spam if p >= {threshold:.3f}); review band [{review_low:.3f}, {threshold:.3f})")

    calibrated = fit_calibrated(X_tr, y_tr)
    plain = fit_calibrated(X_tr, y_tr, augment=False)   # for the robustness comparison only

    # ---- 4. Final, single evaluation on the untouched test set ----------------------------------
    p_te = calibrated.predict_proba(X_te)[:, 1]
    pred_default = (p_te >= 0.5).astype(int)
    pred_tuned = (p_te >= threshold).astype(int)
    m_default, m_tuned = spam_metrics(y_te, pred_default), spam_metrics(y_te, pred_tuned)
    ci = bootstrap_ci(y_te, pred_tuned)

    base = build_pipeline("baseline_clean", "LinearSVM").fit(X_tr, y_tr)
    m_base = spam_metrics(y_te, base.predict(X_te))

    # robustness: obfuscate every test spam message, keep ham untouched
    rng = random.Random(SEED)
    X_obf = [obfuscate(t, rng) if l else t for t, l in zip(X_te, y_te)]
    rob = {"clean_recall_plain": float(recall_score(y_te, plain.predict_proba(X_te)[:, 1] >= threshold)),
           "clean_recall_augmented": m_tuned["recall"],
           "obfuscated_recall_plain": float(recall_score(y_te, plain.predict_proba(X_obf)[:, 1] >= threshold)),
           "obfuscated_recall_augmented": float(recall_score(y_te, calibrated.predict_proba(X_obf)[:, 1] >= threshold))}

    decided = (p_te < review_low) | (p_te >= threshold)
    review = ~decided
    auto_pred = pred_tuned[decided]
    auto_acc = float((auto_pred == y_te.values[decided]).mean())
    summary = {
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "n_test_spam": int(y_te.sum()),
        "best_cv_features": best.features, "best_cv_model": best.model,
        "best_params": {k: float(v) for k, v in grid.best_params_.items()},
        "threshold": threshold, "review_low": review_low, "target_precision": TARGET_PRECISION,
        "test_baseline_original_style": m_base,
        "test_improved_threshold_0.5": m_default,
        "test_improved_tuned_threshold": m_tuned, "test_ci95": ci,
        "obfuscation_robustness": rob,
        "test_average_precision": float(average_precision_score(y_te, p_te)),
        "review_rate": float(review.mean()),
        "auto_decision_accuracy": auto_acc,
        "sklearn_version": sklearn.__version__,
    }
    (OUT / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    print("\nTEST (held out, evaluated once)")
    print(f"  baseline (original style) : {m_base}")
    print(f"  improved @0.5             : {m_default}")
    print(f"  improved @tuned threshold : {m_tuned}\n  95% CI: {ci}")
    print(f"  robustness: {rob}")
    print(f"  review rate {review.mean():.1%} | accuracy on auto-decided {auto_acc:.4f}")

    # ---- 5. Save model ---------------------------------------------------------------------------
    joblib.dump(calibrated, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "config.json").write_text(json.dumps(
        {"threshold": threshold, "review_low": review_low, "features": best.features,
         "model": best.model, "sklearn_version": sklearn.__version__}, indent=2))

    # ---- 6. Plots + error analysis ---------------------------------------------------------------
    plot_cv(cv_df)
    plot_confusion(y_te, pred_tuned, threshold)
    plot_pr(y_te, p_te, threshold, m_tuned)
    plot_scores(p_te, y_te, threshold, review_low)
    plot_top_features(X_tr, y_tr)

    err = pd.DataFrame({"message": X_te.values, "true": y_te.values, "spam_prob": p_te.round(4)})
    err["outcome"] = np.select(
        [(err.true == 0) & (pred_tuned == 1), (err.true == 1) & (pred_tuned == 0), review],
        ["false_positive", "false_negative", "needs_review"], default="ok")
    err[err.outcome != "ok"].sort_values("outcome").to_csv(OUT / "error_analysis.csv", index=False)
    print("\nOutcome counts on test:\n", err.outcome.value_counts().to_string())


def plot_cv(cv_df):
    d = cv_df.copy()
    d["combo"] = d.features + "\n" + d.model
    d = d.sort_values("f1")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#c0504d" if f == "baseline_clean" else "#4a7ebb" for f in d.features]
    ax.barh(d.combo, d.f1, xerr=d.f1_std, color=colors, capsize=3)
    ax.set_xlim(max(0.7, d.f1.min() - 0.03), 1.0)
    ax.set_xlabel("Spam-class F1 (mean +/- std, 5-fold x 2 repeats, training set)")
    ax.set_title("Cross-validated model comparison (red = original-style baseline)")
    plt.tight_layout()
    plt.savefig(OUT / "cv_comparison.png", dpi=130)
    plt.close()


def plot_confusion(y, pred, thr):
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Test confusion matrix (threshold {thr:.2f})")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=130)
    plt.close()


def plot_pr(y, p, thr, m):
    pr, rc, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(6, 4.5))
    plt.plot(rc, pr, color="#4a7ebb", label=f"AP = {average_precision_score(y, p):.3f}")
    plt.scatter([m["recall"]], [m["precision"]], color="crimson", zorder=5,
                label=f"chosen threshold {thr:.2f}")
    plt.axhline(TARGET_PRECISION, ls="--", color="grey", lw=1, label=f"target precision {TARGET_PRECISION}")
    plt.xlabel("Recall (spam)"); plt.ylabel("Precision (spam)"); plt.ylim(0.5, 1.02)
    plt.title("Precision-recall curve (test set)"); plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(OUT / "precision_recall_curve.png", dpi=130)
    plt.close()


def plot_scores(p, y, thr, low):
    plt.figure(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    plt.hist(p[y.values == 0], bins=bins, alpha=0.7, label="ham", color="#4a7ebb")
    plt.hist(p[y.values == 1], bins=bins, alpha=0.7, label="spam", color="#c0504d")
    plt.axvspan(low, thr, color="gold", alpha=0.3, label="review band")
    plt.yscale("log"); plt.xlabel("Predicted spam probability"); plt.ylabel("Messages (log)")
    plt.title("Score distribution on test set"); plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "score_distribution.png", dpi=130)
    plt.close()


def plot_top_features(X_tr, y_tr, k=15):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=3)
    lr = LogisticRegression(C=10, class_weight="balanced", max_iter=2000).fit(vec.fit_transform(X_tr), y_tr)
    names, coef = vec.get_feature_names_out(), lr.coef_[0]
    idx = np.r_[np.argsort(coef)[:k], np.argsort(coef)[-k:]]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(names[idx], coef[idx], color=["#4a7ebb"] * k + ["#c0504d"] * k)
    ax.set_title("Most informative words/phrases (blue = ham, red = spam)")
    plt.tight_layout()
    plt.savefig(OUT / "top_features.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
