import numpy as np, pickle, re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
X=np.load("cache/all_X.npy"); keys=pickle.load(open("cache/all_keys.pkl","rb")); src=pickle.load(open("all_sources.pkl","rb"))
fm=pickle.load(open("fontname_map.pkl","rb"))
from PIL import ImageFont
idx=[i for i,k in enumerate(keys) if src[k] and "myfonts" not in src[k]]
# family = PIL family name (first part); style from name
fam=[];ital=[];wt=[]
for i in idx:
    k=keys[i]; p=fm[k][0][1]
    try: f,s=ImageFont.truetype(p,10).getname()
    except: f,s=k,""
    fam.append(f); sl=s.lower()
    ital.append(int("italic" in sl or "oblique" in sl))
    w=1 if re.search(r"bold|black|heavy|extrabold|ultra",sl) else (-1 if re.search(r"thin|light|hairline",sl) else 0)
    wt.append(w)
Xs=X[idx]; Xs=Xs/np.linalg.norm(Xs,axis=1,keepdims=True); Xs-=Xs.mean(0)
fam=np.array(fam); ital=np.array(ital); wt=np.array(wt)
print("n",len(idx),"italic",ital.sum(),"bold",(wt==1).sum(),"light",(wt==-1).sum())
def probe(Xf,y,groups,name):
    rng=np.random.RandomState(0)
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; n=min(len(pos),len(neg),3000)
    sel=np.concatenate([rng.choice(pos,n,False),rng.choice(neg,n,False)])
    sc=cross_val_score(LogisticRegression(max_iter=3000,C=1),Xf[sel],y[sel],groups=groups[sel],cv=GroupKFold(5),scoring="roc_auc")
    print(f"  {name}: family-grouped CV AUC={sc.mean():.3f}")
probe(Xs,ital,fam,"italic vs upright")
m=wt!=0; probe(Xs[m],(wt[m]==1).astype(int),fam[m],"bold vs light")
# within-family: cos(regular, italic) vs cos(regular, nearest other-family font)
S=Xs/np.linalg.norm(Xs,axis=1,keepdims=True)
from collections import defaultdict
byfam=defaultdict(list)
for j,f in enumerate(fam): byfam[f].append(j)
pairs_i=[];pairs_b=[]
styles=[keys[idx[j]] for j in range(len(idx))]
for f,js in byfam.items():
    reg=[j for j in js if ital[j]==0 and wt[j]==0]
    it=[j for j in js if ital[j]==1 and wt[j]==0]
    bd=[j for j in js if ital[j]==0 and wt[j]==1]
    if reg and it: pairs_i.append((reg[0],it[0]))
    if reg and bd: pairs_b.append((reg[0],bd[0]))
def nnOther(j):
    s=S@S[j]; s[byfam[fam[j]]]=-9; return s.max(), np.sort(s)[::-1]
for name,P in [("regular vs italic",pairs_i),("regular vs bold",pairs_b)]:
    c=[]; nn=[]; rank=[]
    for a,b in P[:400]:
        sab=S[a]@S[b]; m,srt=nnOther(a); c.append(sab); nn.append(m); rank.append((srt>sab).sum())
    c=np.array(c);nn=np.array(nn);rank=np.array(rank)
    print(f"  {name}: n={len(c)} centered-cos same-family={np.median(c):.3f} vs nearest other-family={np.median(nn):.3f}; median # other-family fonts closer than its own {name.split()[-1]} variant = {np.median(rank):.0f} (of {len(S)})")
