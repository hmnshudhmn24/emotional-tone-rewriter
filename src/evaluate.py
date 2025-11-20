from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import config

def evaluate():
    ds=load_dataset('json', data_files='data/processed/dataset_clean.jsonl')['train']
    tok=AutoTokenizer.from_pretrained(config.output_dir)
    m=AutoModelForSeq2SeqLM.from_pretrained(config.output_dir).to(config.device)
    rouge=load_metric('rouge')
    preds=[]; refs=[]
    for it in ds:
        enc=tok(it['input'],return_tensors='pt').to(config.device)
        out=m.generate(**enc, max_length=config.max_target_length)
        preds.append(tok.decode(out[0],skip_special_tokens=True))
        refs.append(it['target'])
    print(rouge.compute(predictions=preds,references=refs))

if __name__=='__main__': evaluate()
