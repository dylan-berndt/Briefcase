import pickle, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from collections import Counter
G=pickle.load(open("glyphemb.pkl","rb")); lab,_=pickle.load(open("labels.pkl","rb"))
keys=pickle.load(open("cache/all_keys.pkl","rb")); X=np.load("cache/all_X.npy"); ki={k:i for i,k in enumerate(keys)}
ks=sorted(G); rng=np.random.RandomState(0); p=rng.permutation(len(ks)); te=set(ks[i] for i in p[:1200])
tr=[k for k in ks if k not in te]; te=sorted(te)
cnt=Counter(t for k in tr for t in lab[k][1]); tags=[t for t,c in cnt.most_common(150) if sum(t in lab[k][1] for k in te)>=10]
reps={"all.json (case-mixed)":{k:X[ki[k]] for k in ks},"lowercase-only (fixed)":{k:G[k][:26].astype(np.float32).mean(0) for k in ks}}
for name,E in reps.items():
    A=np.stack([E[k] for k in tr]); mu=A.mean(0); A=normalize(A-mu); B=normalize(np.stack([E[k] for k in te])-mu); au=[]
    for t in tags:
        y=np.array([t in lab[k][1] for k in tr]); yt=np.array([t in lab[k][1] for k in te])
        au.append(roc_auc_score(yt,LogisticRegression(max_iter=1000,class_weight="balanced").fit(A,y).decision_function(B)))
    print(f"  {name:24s} mean tag AUC over {len(tags)} tags = {np.mean(au):.4f}")
