import json, pickle, numpy as np, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
keys=pickle.load(open("cache/all_keys.pkl","rb")); X=np.load("cache/all_X.npy"); ki={k:i for i,k in enumerate(keys)}
src=pickle.load(open("all_sources.pkl","rb")); lab,_=pickle.load(open("labels.pkl","rb"))
Q=json.load(open("results/fontQueries.json"))
F=[k for k in Q if k in ki and src[k]=={"myfonts"} and len(Q[k])>=2]
rng=np.random.RandomState(0); perm=rng.permutation(len(F)); nte=4000
te=[F[i] for i in perm[:nte]]; tr=[F[i] for i in perm[nte:]]
hold={}; teDoc={}
for k in te:
    qs=Q[k]; i=rng.randint(len(qs)); hold[k]=qs[i]; teDoc[k]=" . ".join(qs[:i]+qs[i+1:])
trDoc=[" . ".join(Q[k]) for k in tr]
vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True).fit(trDoc+list(teDoc.values()))
T_tr=normalize(vec.transform(trDoc)); H=normalize(vec.transform([hold[k] for k in te]))
def evalGallery(G,name):
    G=normalize(G); S=(H@G.T); S=S.toarray() if hasattr(S,"toarray") else np.asarray(S)
    t=np.diag(S); r=(S>t[:,None]).sum(1)
    print(f"  {name:45s} R@1={np.mean(r<1):.3f} R@10={np.mean(r<10):.3f} R@100={np.mean(r<100):.3f} medRank={np.median(r)+1:.0f}/{len(te)}")
    return r
def knnPred(Vtr,Vte,k=30,temp=None):
    Vtr=normalize(Vtr); Vte=normalize(Vte); S=Vte@Vtr.T
    idx=np.argpartition(-S,k,axis=1)[:,:k]
    rows=[]
    import scipy.sparse as sp
    W=np.zeros_like(S)
    for i in range(len(Vte)):
        s=S[i,idx[i]]; w=np.exp((s-s.max())/temp) if temp else np.ones_like(s)
        W[i,idx[i]]=w/w.sum()
    return sp.csr_matrix(W)@T_tr
def feats(K,which="all"):
    return np.stack([X[ki[k]] for k in K])
print(f"MyFonts only, V1 queries. gallery={len(te)} held-out fonts; kNN memory={len(tr)} train fonts")
evalGallery(vec.transform([teDoc[k] for k in te]),"ORACLE text->text (own other queries)")
# tag-jaccard kNN (oracle 'perfect tag-similarity' visual space)
tags=lambda k: lab[k][1]
allt=sorted(set().union(*[tags(k) for k in tr+te])); ti={t:i for i,t in enumerate(allt)}
def tagmat(K):
    import scipy.sparse as sp
    r=[];c=[]
    for i,k in enumerate(K):
        for t in tags(k): r.append(i);c.append(ti[t])
    return sp.csr_matrix((np.ones(len(r)),(r,c)),shape=(len(K),len(allt)))
from sklearn.feature_extraction.text import TfidfTransformer
tt=TfidfTransformer().fit(tagmat(tr)); Ttr=tt.transform(tagmat(tr)).toarray(); Tte=tt.transform(tagmat(te)).toarray()
for k in [10,30]:
    evalGallery(knnPred(Ttr,Tte,k),f"kNN via TRUE TAGS (k={k}) [label oracle]")
Vtr=feats(tr); Vte=feats(te); mu=Vtr.mean(0)
for k in [10,30,100]:
    evalGallery(knnPred(Vtr-mu,Vte-mu,k),f"kNN via all.json ViT (k={k})")
rv=np.random.RandomState(1).randn(len(tr)+len(te),64)
evalGallery(knnPred(rv[:len(tr)],rv[len(tr):],30),"kNN via RANDOM features (k=30) [prior]")
auc=pickle.load(open("tagauc_vit.pkl","rb"))
def restricted(pred,name):
    keep=[t for t in allt if t in auc and pred(auc[t])]
    kk=np.array([ti[t] for t in keep])
    A=tagmat(tr)[:,kk]; B=tagmat(te)[:,kk]; tt2=TfidfTransformer().fit(A)
    evalGallery(knnPred(tt2.transform(A).toarray(),tt2.transform(B).toarray(),30),f"kNN via TRUE TAGS, {name} ({len(keep)} tags)")
restricted(lambda a:a>=0.85,"visual only AUC>=.85")
restricted(lambda a:a>=0.75,"AUC>=.75")
restricted(lambda a:a<0.75,"non-visual only AUC<.75")
