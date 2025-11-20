from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import config

def load_tokenizer(m=None):
    return AutoTokenizer.from_pretrained(m or config.model_name)

def load_model(m=None):
    return AutoModelForSeq2SeqLM.from_pretrained(m or config.model_name)

def save_model_and_tokenizer(model, tok, out):
    model.save_pretrained(out); tok.save_pretrained(out)
