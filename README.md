# 🎭 Emotional Tone Rewriter

A Transformer-based NLP model that rewrites sentences into **different emotional tones** such as:

- friendly  
- angry  
- sad  
- corporate  
- romantic  
- sarcastic  
- motivational  
- formal  
- cheerful  

Built using **T5-small**, this project includes a complete pipeline: preprocessing, training, evaluation, inference, FastAPI server, and Gradio UI.



# 🚀 Features

✓ Rewrite any sentence into a selected emotional tone  
✓ Fully reproducible training pipeline (PyTorch + HuggingFace)  
✓ Clean dataset preprocessing (`CSV → JSONL`)  
✓ Gradio UI for interactive demos  
✓ FastAPI server for deployment  
✓ Evaluation notebook with ROUGE metrics  
✓ Apache 2.0 licensed (commercial-friendly)  
✓ Production-grade project structure  



# 📁 Project Structure

```
emotional-tone-rewriter/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── src/
│   ├── config.py
│   ├── dataset_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── model_utils.py
├── app/
│   ├── api.py
│   └── ui.py
├── model/
├── notebooks/
├── huggingface/
├── requirements.txt
├── LICENSE
└── README.md
```



# 📦 Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


# 🔄 Dataset Preprocessing

```bash
python -m src.dataset_preprocessing --input data/raw/dataset_raw.csv --output data/processed/dataset_clean.jsonl
```



# 🏋️ Train Model

```bash
python -m src.train
```



# 🧪 Evaluate

```bash
python -m src.evaluate
```



# 🤖 Inference Example

```python
from src.inference import rewrite_tone
print(rewrite_tone("I can't come today.", "sarcastic"))
```



# 🌐 FastAPI Server

```bash
uvicorn app.api:app --reload --port 7860
```



# 🎨 Gradio UI

```bash
python app/ui.py
```


# 📄 License
Apache License 2.0
