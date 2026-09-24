import numpy as np, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
X=np.load("cache/all_X.npy"); keys=pickle.load(open("cache/all_keys.pkl","rb"))
src=pickle.load(open("all_sources.pkl","rb"))
lab=[]; idx=[]
for i,k in enumerate(keys):
    s=src[k]
    if len(s)==1: lab.append(list(s)[0]); idx.append(i)
X=X[idx]; y=np.array(lab)
Xn=X/np.linalg.norm(X,axis=1,keepdims=True); Xc=Xn-Xn.mean(0)
print({c:(y==c).sum() for c in set(y)})
# balanced subsample
rng=np.random.RandomState(0); sel=np.concatenate([rng.choice(np.where(y==c)[0],3500,replace=False) for c in ["myfonts","dafont","google"]])
clf=LogisticRegression(max_iter=3000,C=1.0)
sc=cross_val_score(clf,Xc[sel],y[sel],cv=StratifiedKFold(5,shuffle=True,random_state=0))
print("linear probe source acc (balanced 3-way, chance .333):",sc.mean())
# myfonts vs rest binary on all
yb=(y=="myfonts")
sc=cross_val_score(LogisticRegression(max_iter=3000),Xc[sel],yb[sel],cv=5)
print("myfonts-vs-other acc (chance .667):",sc.mean())
# kNN source purity: fraction of 10-NN with same source
S=Xc[sel]/np.linalg.norm(Xc[sel],axis=1,keepdims=True)
sim=S@S.T; np.fill_diagonal(sim,-9)
nn=np.argsort(-sim,1)[:,:10]
same=(y[sel][nn]==y[sel][:,None]).mean()
print("10-NN same-source fraction (balanced, chance .333):",same)
