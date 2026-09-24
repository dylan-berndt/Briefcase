import pickle, numpy as np, random, pandas as pd, re
from render import *; from model import *
from sklearn.metrics import roc_auc_score
torch.set_num_threads(4)
L="abcdefghijklmnopqrstuvwxyz"
keys=pickle.load(open("cache/all_keys.pkl","rb")); src=pickle.load(open("all_sources.pkl","rb")); fm=pickle.load(open("fontname_map.pkl","rb"))
lab,_=pickle.load(open("labels.pkl","rb"))
D=[k for k in keys if src[k]=={"dafont"} and len(fm[k])==1]
random.seed(0); D=random.sample(D,1500)
m=load(); A=[];B=[];ok=[]
for n,k in enumerate(D):
    p=fm[k][0][1]
    try:
        a=[metricRender(p,c) for c in L]; b=[myfontsStyleFromFont(p,c) for c in L]
    except Exception: continue
    if any(x is None for x in a+b): continue
    Ea=embedGlyphs(m,a); Eb=embedGlyphs(m,b)
    A.append(Ea/np.linalg.norm(Ea,axis=1,keepdims=True)); B.append(Eb/np.linalg.norm(Eb,axis=1,keepdims=True)); ok.append(k)
    if n%200==0: print(n,flush=True)
A=np.stack(A).astype(np.float32); B=np.stack(B).astype(np.float32)
pickle.dump((ok,A,B),open("pipeline_pairs.pkl","wb")); print("saved",len(ok))
