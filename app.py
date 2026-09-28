"""Streamlit demo:  streamlit run app.py"""
import streamlit as st

from src.explain import explain
from src.inference import classify, load

st.set_page_config(page_title="SMS Spam Filter", page_icon="📩")
st.title("📩 SMS Spam / Ham Classifier")
st.caption("Character n-gram TF-IDF + calibrated Linear SVM, with a human-review band for uncertain messages.")

_, cfg = load()
text = st.text_area("Paste an SMS message", height=120,
                    placeholder="Congratulations! You've won a free prize, text WIN to 80086")
if st.button("Classify") and text.strip():
    r = classify([text])[0]
    p = r["spam_probability"]
    if r["decision"] == "SPAM":
        st.error(f"🚫 SPAM  (p = {p:.1%})")
    elif r["decision"] == "HAM":
        st.success(f"✅ HAM  (p(spam) = {p:.1%})")
    else:
        st.warning(f"⚠️ NEEDS REVIEW  (p(spam) = {p:.1%}) — the model is unsure.")
    st.progress(min(max(p, 0.0), 1.0))
    top = explain(text)
    if top:
        st.subheader("Why?")
        st.caption("Words that changed the spam log-odds most when removed (+ = pushes towards spam).")
        st.table({"word": [w for w, _ in top], "effect (log-odds)": [f"{d:+.3f}" for _, d in top]})
with st.expander("How decisions are made"):
    st.write(f"SPAM if p ≥ {cfg['threshold']:.3f}; HAM if p < {cfg['review_low']:.3f}; otherwise REVIEW. "
             "The threshold was tuned on training-set cross-validation to hit a target spam precision.")
