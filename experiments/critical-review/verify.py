import pickle, numpy as np, random, json
from render import *; from model import *
X=np.load("cache/all_X.npy"); keys=pickle.load(open("cache/all_keys.pkl","rb")); ki={k:i for i,k in enumerate(keys)}
src=pickle.load(open("all_sources.pkl","rb")); fm=pickle.load(open("fontname_map.pkl","rb"))
m=load(); L="abcdefghijklmnopqrstuvwxyz"
random.seed(0)
G=[k for k in keys if src[k]=={"google"}]; D=[k for k in keys if src[k]=={"dafont"}]; M=[k for k in keys if src[k]=={"myfonts"}]
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
for name,group in [("google",G),("dafont",D)]:
    cs=[]
    for k in random.sample(group,15):
        p=fm[k][0][1]; imgs=[metricRender(p,c) for c in L]
        if any(i is None for i in imgs): continue
        cs.append(cos(fontVec(embedGlyphs(m,imgs)),X[ki[k]]))
    print(name,"reproduction cos:",np.round(cs,4))
cs=[]
for k in random.sample(M,15):
    imgs=[rochesterFile(f"dataset/fontimage/{k}_{c}.png") for c in L]
    cs.append(cos(fontVec(embedGlyphs(m,imgs)),X[ki[k]]))
print("myfonts reproduction cos:",np.round(cs,4))
