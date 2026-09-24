import pickle, numpy as np, random
from render import *; from model import *
X=np.load("cache/all_X.npy"); keys=pickle.load(open("cache/all_keys.pkl","rb")); ki={k:i for i,k in enumerate(keys)}
src=pickle.load(open("all_sources.pkl","rb")); fm=pickle.load(open("fontname_map.pkl","rb"))
m=load(); L="abcdefghijklmnopqrstuvwxyz"; random.seed(0)
def nrm(E): return E/np.linalg.norm(E,axis=1,keepdims=True)
def fit(lo,up,target):
    A=(up-lo).T; b=26*target-lo.sum(0)
    beta=np.linalg.lstsq(A,b,rcond=None)[0]; bb=(beta>0.5).astype(float)
    v=(lo+bb[:,None]*(up-lo)).mean(0); return bb, float(v@target/np.linalg.norm(v)/np.linalg.norm(target))
M=[k for k in keys if src[k]=={"myfonts"}]; D=[k for k in keys if src[k]=={"dafont"}]
res=[]
for k in random.sample(M,12):
    lo=nrm(embedGlyphs(m,[rochesterFile(f"dataset/fontimage/{k}_{c}.png") for c in L]))
    up=nrm(embedGlyphs(m,[rochesterFile(f"dataset/fontimage/{k}_{c.upper()*2}.png") for c in L]))
    bb,c=fit(lo,up,X[ki[k]]); res.append((round(c,4),int(bb.sum())))
print("myfonts best case-mix cos, #uppercase used:",res)
res=[]
for k in random.sample(D,12):
    p=fm[k][0][1]; lo=[metricRender(p,c) for c in L]; up=[metricRender(p,c.upper()) for c in L]
    if any(i is None for i in lo+up): continue
    bb,c=fit(nrm(embedGlyphs(m,lo)),nrm(embedGlyphs(m,up)),X[ki[k]]); res.append((round(c,4),int(bb.sum())))
print("dafont best case-mix cos, #uppercase used:",res)
