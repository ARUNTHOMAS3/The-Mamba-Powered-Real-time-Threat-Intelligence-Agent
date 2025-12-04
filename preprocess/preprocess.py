# preprocess/preprocess.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, os, json, numpy as np
from utils.utils import mkdirp

def generate_synthetic(out_path, n_samples=200, seq_len=128):
    mkdirp(os.path.dirname(out_path))
    data = []
    for i in range(n_samples):
        label = 1 if np.random.rand() < 0.2 else 0
        x_log = np.random.normal(0,1,(seq_len,32))
        x_text = np.random.normal(0,1,(seq_len,64))
        x_cve = np.random.normal(0,1,(seq_len,16))
        if label==1:
            idx = np.random.randint(seq_len//2, seq_len-4)
            x_log[idx:idx+4] += np.random.normal(3.0,0.5,(4,32))
            x_text[idx:idx+4] += np.random.normal(2.0,0.3,(4,64))
        data.append({"x_log":x_log.tolist(),"x_text":x_text.tolist(),"x_cve":x_cve.tolist(),"label":int(label)})
    json.dump(data, open(out_path,"w"))
    print(f"Saved synthetic dataset to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed/synth_small.json")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seq", type=int, default=128)
    args = parser.parse_args()
    generate_synthetic(args.out, n_samples=args.n, seq_len=args.seq)
