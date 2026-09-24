import json, pickle, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
keys=pickle.load(open("cache/all_keys.pkl","rb")); X=np.load("cache/all_X.npy"); ki={k:i for i,k in enumerate(keys)}
src=pickle.load(open("all_sources.pkl","rb")); lab,_=pickle.load(open("labels.pkl","rb")); auc=pickle.load(open("tagauc_vit.pkl","rb"))
Q=json.load(open("results/fontQueries.json"))
F=[k for k in Q if k in ki and src[k]=={"myfonts"} and len(Q[k])>=2]; N=len(F)
rng=np.random.RandomState(0); hold=[];own=[]
for k in F:
    qs=Q[k]; i=rng.randint(len(qs)); hold.append(qs[i]); own.append(" . ".join(qs[:i]+qs[i+1:]))
vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True).fit(own)
D=normalize(vec.transform(own)); H=normalize(vec.transform(hold))
def ev(G,name):
    G=normalize(sp.csr_matrix(G)); r=np.empty(N,int)
    for s in range(0,N,1000):
        S=(H[s:s+1000]@G.T).toarray(); t=S[np.arange(S.shape[0]),np.arange(s,s+S.shape[0])]; r[s:s+S.shape[0]]=(S>t[:,None]).sum(1)
    print(f"  {name:44s} R@1={np.mean(r<1):.3f} R@10={np.mean(r<10):.3f} R@100={np.mean(r<100):.3f} medRank={np.median(r)+1:.0f}/{N}",flush=True)
def looKnn(V,k=30):
    V=normalize(V); rows=[];cols=[]
    for s in range(0,N,1000):
        S=V[s:s+1000]@V.T; S=S.toarray() if sp.issparse(S) else S
        S[np.arange(S.shape[0]),np.arange(s,s+S.shape[0])]=-9
        idx=np.argpartition(-S,k,1)[:,:k]; rows.append(np.repeat(np.arange(s,s+S.shape[0]),k)); cols.append(idx.ravel())
    W=sp.csr_matrix((np.full(N*k,1/k),(np.concatenate(rows),np.concatenate(cols))),shape=(N,N))
    return W@D
print(f"Full-scale MyFonts leave-one-out, V1 captions, gallery={N}")
ev(D,"ORACLE text->text (own other captions)")
tags=sorted(set().union(*[lab[k][1] for k in F])); ti={t:i for i,t in enumerate(tags)}
r=[];c=[]
for i,k in enumerate(F):
    for t in lab[k][1]: r.append(i); c.append(ti[t])
TM=sp.csr_matrix((np.ones(len(r)),(r,c)),shape=(N,len(tags)))
def tagsub(pred):
    cols=np.array([ti[t] for t in tags if t in auc and pred(auc[t])]); M=TM[:,cols]; return TfidfTransformer().fit_transform(M)
ev(looKnn(TfidfTransformer().fit_transform(TM)),"kNN via TRUE TAGS (all)")
ev(looKnn(tagsub(lambda a:a>=0.85)),"kNN via TRUE VISUAL TAGS (AUC>=.85)")
V=np.stack([X[ki[k]] for k in F]); V=V-V.mean(0)
ev(looKnn(V),"kNN via ViT all.json")
