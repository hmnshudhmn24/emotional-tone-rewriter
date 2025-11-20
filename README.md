# Emotional Tone Rewriter

Rewrite sentences into a specified emotional tone using a T5 seq2seq model.

## Quick start

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Preprocess data:
```bash
python -m src.dataset_preprocessing --input data/raw/dataset_raw.csv --output data/processed/dataset_clean.jsonl
```

3. Train:
```bash
python -m src.train
```

4. API:
```bash
uvicorn app.api:app --reload
```

5. Gradio:
```bash
python app/ui.py
```
