.PHONY: install train test lint app api
install: ; pip install -r requirements.txt
train:   ; python train.py
test:    ; pytest -q
lint:    ; ruff check .
app:     ; streamlit run app.py
api:     ; uvicorn api:app --reload
