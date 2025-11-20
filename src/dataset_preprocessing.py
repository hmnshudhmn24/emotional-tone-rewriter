import argparse, csv, json
from pathlib import Path

def csv_to_jsonl(input_csv, output_jsonl):
    p_in = Path(input_csv); p_out = Path(output_jsonl)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    with p_in.open() as fin, p_out.open('w') as fout:
        r = csv.DictReader(fin)
        for row in r:
            s = row['sentence']; t = row['tone']; out = row['target']
            fout.write(json.dumps({"input":f"rewrite: {t} | {s}","target":out})+"
")

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input'); p.add_argument('--output')
    a=p.parse_args()
    csv_to_jsonl(a.input,a.output)
