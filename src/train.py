import os, random, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from src.config import config

def set_seed(s):
    random.seed(s); os.environ['PY']=str(s)
    import numpy as np; np.random.seed(s); torch.manual_seed(s)

def preprocess(ex, tok):
    m = tok(ex['input'], truncation=True, max_length=config.max_input_length)
    with tok.as_target_tokenizer():
        lbl=tok(ex['target'], truncation=True, max_length=config.max_target_length)
    m['labels']=lbl['input_ids']; return m

def main():
    set_seed(config.seed)
    ds = load_dataset('json', data_files='data/processed/dataset_clean.jsonl')['train']
    ds = ds.train_test_split(test_size=0.1)
    tok = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    train = ds['train'].map(lambda ex: preprocess(ex,tok), batched=True)
    eval = ds['test'].map(lambda ex: preprocess(ex,tok), batched=True)
    args = Seq2SeqTrainingArguments(output_dir=config.output_dir, evaluation_strategy='epoch', 
        per_device_train_batch_size=config.train_batch_size, per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.lr, num_train_epochs=config.epochs)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train, eval_dataset=eval, tokenizer=tok,
                             data_collator=DataCollatorForSeq2Seq(tok,model))
    trainer.train(); trainer.save_model(config.output_dir); tok.save_pretrained(config.output_dir)

if __name__=='__main__': main()
