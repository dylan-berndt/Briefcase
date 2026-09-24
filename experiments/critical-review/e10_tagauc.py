import numpy as np, pickle, sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from collections import Counter
lab,_=pickle.load(open("labels.pkl","rb"))
name,path=sys.argv[1],sys.argv[2]
if path=="all.json":
    keys=pickle.load(open("cache/all_keys.pkl","rb")); X=np.load("cache/all_X.npy"); E=dict(zip(keys,X))
else: E=pickle.load(open(path,"rb"))
tr=[l.strip() for l in open("dataset/fontset/trainset") if l.strip() in E]
te=[l.strip() for l in open("dataset/fontset/testset") if l.strip() in E]+[l.strip() for l in open("dataset/fontset/valset") if l.strip() in E]
if len(sys.argv)>3:  # restrict to a common key set for fair comparison
    common=pickle.load(open(sys.argv[3],"rb")); tr=[k for k in tr if k in common]; te=[k for k in te if k in common]
cnt=Counter(t for k in tr for t in lab[k][1]); tags=[t for t,c in cnt.most_common() if c>=40][:400]
Xtr=np.stack([E[k] for k in tr]).astype(np.float64); mu=Xtr.mean(0); Xtr=normalize(Xtr-mu); Xte=normalize(np.stack([E[k] for k in te])-mu)
auc={}
for t in tags:
    y=np.array([t in lab[k][1] for k in tr]); yt=np.array([t in lab[k][1] for k in te])
    if yt.sum()<5: continue
    clf=LogisticRegression(max_iter=1000,C=1.0,class_weight="balanced").fit(Xtr,y)
    auc[t]=roc_auc_score(yt,clf.decision_function(Xte))
pickle.dump(auc,open(f"tagauc_{name}.pkl","wb"))
a=np.array(list(auc.values()))
print(f"{name}: train={len(tr)} test={len(te)} tags={len(a)} mean AUC={a.mean():.3f} median={np.median(a):.3f}")
s=sorted(auc.items(),key=lambda x:-x[1])
print(" top:",[(t,round(v,2)) for t,v in s[:15]]); print(" bottom:",[(t,round(v,2)) for t,v in s[-15:]])
