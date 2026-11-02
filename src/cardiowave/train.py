"""Run the complete synthetic baseline and noise robustness benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .data import generate, group_split
from .features import matrix

def evaluate(y, pred):
    return {
        "accuracy": float(accuracy_score(y,pred)),
        "macro_f1": float(f1_score(y,pred,average="macro",labels=[0,1,2],zero_division=0)),
        "confusion_matrix": confusion_matrix(y,pred,labels=[0,1,2]).tolist()
    }

def run(seed=20261030, subjects=60, windows_per_class=4, sample_rate=100, seconds=8):
    signals, labels, groups = generate(seed,subjects,windows_per_class,sample_rate,seconds)
    tr,va,te = group_split(groups,seed)
    X = matrix(signals,sample_rate)
    model = RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=1,
                                   class_weight="balanced")
    model.fit(X[tr],labels[tr])
    validation = evaluate(labels[va],model.predict(X[va]))
    clean = evaluate(labels[te],model.predict(X[te]))
    noisy_signals = signals[te]+np.random.default_rng(seed+1).normal(
        0,.2,signals[te].shape)
    noisy = evaluate(labels[te],model.predict(matrix(noisy_signals,sample_rate)))
    return {"seed":seed,"subjects":subjects,"sample_count":len(labels),
            "split_samples":{"train":len(tr),"validation":len(va),"test":len(te)},
            "validation":validation,"test":{"clean":clean,"noise_stress":noisy},
            "disclaimer":"Synthetic research demonstration only; not for clinical use."}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=20261030)
    parser.add_argument("--subjects",type=int,default=60)
    parser.add_argument("--windows-per-class",type=int,default=4)
    parser.add_argument("--sample-rate",type=int,default=100)
    parser.add_argument("--seconds",type=int,default=8)
    parser.add_argument("--output",default="outputs/metrics.json")
    args=parser.parse_args()
    result=run(args.seed,args.subjects,args.windows_per_class,args.sample_rate,args.seconds)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["test"],indent=2))

if __name__=="__main__":
    main()
