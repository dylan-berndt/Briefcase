import json, pickle, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
C=pickle.load(open("clip_myfonts.pkl","rb")); V=pickle.load(open("vit_on_clipkeys.pkl","rb"))
Q=json.load(open("results/fontQueries.json"))
F=sorted(k for k in C if k in V and k in Q and len(Q[k])>=2)
rng=np.random.RandomState(0); perm=rng.permutation(len(F)); nte=2000
te=[F[i] for i in perm[:nte]]; tr=[F[i] for i in perm[nte:]]
hold=[];teDoc=[]
for k in te:
    qs=Q[k]; i=rng.randint(len(qs)); hold.append(qs[i]); teDoc.append(" . ".join(qs[:i]+qs[i+1:]))
trDoc=[" . ".join(Q[k]) for k in tr]
vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True).fit(trDoc+teDoc)
T=normalize(vec.transform(trDoc)); H=normalize(vec.transform(hold))
def ev(G,name):
    G=normalize(G); S=H@G.T; S=S.toarray() if hasattr(S,"toarray") else np.asarray(S)
    t=np.diag(S); r=(S>t[:,None]).sum(1)
    print(f"  {name:32s} R@1={np.mean(r<1):.3f} R@10={np.mean(r<10):.3f} R@100={np.mean(r<100):.3f} medRank={np.median(r)+1:.0f}/{len(te)}")
def knn(E,k=30):
    A=np.stack([E[x] for x in tr]).astype(np.float64); B=np.stack([E[x] for x in te]).astype(np.float64); mu=A.mean(0)
    A=normalize(A-mu); B=normalize(B-mu); S=B@A.T; idx=np.argpartition(-S,k,1)[:,:k]
    W=np.zeros_like(S); W[np.arange(len(B))[:,None],idx]=1/k; return sp.csr_matrix(W)@T
print(f"MyFonts subset: gallery={len(te)}, memory={len(tr)}")
ev(vec.transform(teDoc),"ORACLE text->text")
ev(knn(V),"kNN ViT (all.json)"); ev(knn(C),"kNN CLIP B/16")
Cat={k:np.concatenate([V[k]/np.linalg.norm(V[k]),C[k]/np.linalg.norm(C[k])]) for k in F}
ev(knn(Cat),"kNN ViT+CLIP concat")
