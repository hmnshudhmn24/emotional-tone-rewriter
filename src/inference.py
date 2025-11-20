from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import config

_cache={}

def rewrite_tone(s, tone, d=None):
    d=d or config.output_dir
    if d not in _cache:
        tok = AutoTokenizer.from_pretrained(d)
        m = AutoModelForSeq2SeqLM.from_pretrained(d).to(config.device)
        _cache[d]=(tok,m)
    tok,m=_cache[d]
    inp=tok(f"rewrite: {tone} | {s}", return_tensors='pt').to(config.device)
    out=m.generate(**inp, max_length=config.max_target_length)
    return tok.decode(out[0], skip_special_tokens=True)
