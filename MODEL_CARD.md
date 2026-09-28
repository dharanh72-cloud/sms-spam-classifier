# Model card — SMS Spam / Ham classifier

**Model**: character n-gram (2–5, `char_wb`) TF-IDF → class-weighted Linear SVM (C=1), sigmoid-calibrated (5-fold),
trained with obfuscation-augmented spam. Output: `spam_probability` and a decision **SPAM / HAM / REVIEW**.

**Intended use**: first-pass filtering of short English SMS, with uncertain messages routed to a human or rule check.
**Not intended for**: fully automatic blocking without review, non-English text, email, or modern app-based scams (WhatsApp/OTP/delivery)
without retraining on representative data.

## Data
UCI SMS Spam Collection (≈2004–2012, UK/Singapore-heavy English). After removing exact duplicates: 5,158 messages, 12.4% spam.
Near-duplicate templates are grouped so they never straddle train/test. HTML entities are unescaped.

## Evaluation (held-out grouped test set, 1,032 messages / 128 spam, evaluated once)
| Metric (spam class) | Value | 95% bootstrap CI |
|---|---|---|
| Precision | 0.976 | 0.945 – 1.000 |
| Recall | 0.961 | 0.926 – 0.992 |
| F1 | 0.969 | 0.945 – 0.988 |

* Average precision 0.983. 3 false positives, 5 false negatives, 53 messages (5.4%) sent to REVIEW.
* Accuracy on auto-decided (non-REVIEW) messages: 99.5%.
* Recall on obfuscated spam (`fr33`, `w.i.n`, `pr ize`): 0.984 (0.938 without augmentation).
* The threshold targets 0.98 spam precision on out-of-fold predictions; the test set landed at 0.976 — within sampling noise.

## Known limitations & risks
* **Small test set**: one spam message ≈ 0.8 points of recall; differences under ~0.01 are noise.
* **Domain drift**: old, English, SMS-only data; expect lower accuracy on modern phishing and other languages.
* **Ambiguous cases**: legitimate bank/security notices and "send free SMS" invitations resemble spam; chatty premium-rate spam looks like ham.
* **Adversarial spam**: only simple obfuscation is covered; a determined attacker can adapt.
* **Explanations** are word-occlusion approximations, not causal proofs.
* **Fairness/privacy**: messages may contain personal data; do not log or share raw messages without consent.

## Maintenance
Retrain (`python train.py`) when the scikit-learn version changes or drift is observed; the saved model is tied to the
scikit-learn version recorded in `models/config.json`.
