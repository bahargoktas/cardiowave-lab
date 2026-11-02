"""Synthetic waveform experiment: group split, RF and CNN, noisy evaluation."""
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .data import generate, group_split
from .features import matrix
from .model import WaveCNN

def evaluate(y, pred):
    return {"accuracy":float(accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,average="macro",labels=[0,1,2],zero_division=0)),
            "confusion_matrix":confusion_matrix(y,pred,labels=[0,1,2]).tolist()}

def cnn_predict(model,x):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x[:,None,:].astype(np.float32))).argmax(1).numpy()

def train_cnn(x,y,x_val,y_val,seed=42,epochs=10,batch_size=64):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model=WaveCNN()
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(torch.from_numpy(x[:,None,:].astype(np.float32)),
        torch.from_numpy(y.astype(np.int64))),batch_size=batch_size,shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    best=-1.0
    state=None
    history=[]
    for epoch in range(1,epochs+1):
        model.train()
        losses=[]
        for xb,yb in loader:
            opt.zero_grad()
            loss=nn.functional.cross_entropy(model(xb),yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        f1=float(f1_score(y_val,cnn_predict(model,x_val),average="macro",zero_division=0))
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_macro_f1":f1})
        if f1>best:
            best=f1
            state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model,history

def run(seed=20261030,subjects=60,windows_per_class=4,sample_rate=100,seconds=8,
        cnn_epochs=15,noise_std=.2):
    signals,labels,groups=generate(seed,subjects,windows_per_class,sample_rate,seconds)
    tr,va,te=group_split(groups,seed)
    features=matrix(signals,sample_rate)
    best_rf=None
    best_f1=-1
    for leaf in (1,2,4):
        candidate=RandomForestClassifier(n_estimators=180,min_samples_leaf=leaf,
                    random_state=seed,n_jobs=1)
        candidate.fit(features[tr],labels[tr])
        f1=f1_score(labels[va],candidate.predict(features[va]),average="macro",zero_division=0)
        if f1>best_f1:
            best_f1=f1
            best_rf=candidate
    mu=float(signals[tr].mean())
    sigma=max(float(signals[tr].std()),1e-8)
    scaled=((signals-mu)/sigma).astype(np.float32)
    cnn,history=train_cnn(scaled[tr],labels[tr],scaled[va],labels[va],
                          seed=seed,epochs=cnn_epochs)
    rng=np.random.default_rng(seed+9001)
    noisy=signals[te]+rng.normal(0,noise_std,signals[te].shape)
    clean_rf=evaluate(labels[te],best_rf.predict(features[te]))
    noisy_rf=evaluate(labels[te],best_rf.predict(matrix(noisy,sample_rate)))
    clean_cnn=evaluate(labels[te],cnn_predict(cnn,scaled[te]))
    noisy_cnn=evaluate(labels[te],cnn_predict(cnn,((noisy-mu)/sigma).astype(np.float32)))
    return {"seed":seed,"subjects":subjects,"sample_count":len(labels),
        "split_samples":{"train":len(tr),"validation":len(va),"test":len(te)},
        "validation":{"random_forest_macro_f1":float(best_f1),
                      "cnn_best_macro_f1":max(h["validation_macro_f1"] for h in history)},
        "test":{"random_forest":clean_rf,"cnn_1d":clean_cnn},
        "noise_stress":{"random_forest":noisy_rf,"cnn_1d":noisy_cnn},
        "cnn_history":history,
        "disclaimer":"Entirely synthetic signals; research only; not for clinical use."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=20261030)
    p.add_argument("--subjects",type=int,default=60)
    p.add_argument("--windows-per-class",type=int,default=4)
    p.add_argument("--sample-rate",type=int,default=100)
    p.add_argument("--seconds",type=int,default=8)
    p.add_argument("--cnn-epochs",type=int,default=15)
    p.add_argument("--noise-std",type=float,default=.2)
    p.add_argument("--output",default="outputs/metrics.json")
    a=p.parse_args()
    result=run(a.seed,a.subjects,a.windows_per_class,a.sample_rate,
               a.seconds,a.cnn_epochs,a.noise_std)
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result["test"],indent=2))

if __name__=="__main__":
    main()
