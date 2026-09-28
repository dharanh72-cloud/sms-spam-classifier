"""REST API:  uvicorn api:app --reload      then open http://127.0.0.1:8000/docs"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.explain import explain
from src.inference import classify, load

app = FastAPI(title="SMS Spam Filter", version="2.0")


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    explain: bool = False


class Batch(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)


@app.get("/health")
def health():
    _, cfg = load()
    return {"status": "ok", "threshold": cfg["threshold"], "review_low": cfg["review_low"]}


@app.post("/classify")
def classify_one(m: Message):
    result = classify([m.text])[0]
    if m.explain:
        result["top_words"] = [{"word": w, "effect": d} for w, d in explain(m.text)]
    return result


@app.post("/classify/batch")
def classify_batch(b: Batch):
    return classify(b.texts)
