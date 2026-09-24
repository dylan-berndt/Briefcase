exec(open("e17_fullscale.py").read().split('print(f"Full-scale')[0])
tags=sorted(set().union(*[lab[k][1] for k in F])); ti={t:i for i,t in enumerate(tags)}
r=[];c=[]
for i,k in enumerate(F):
    for t in lab[k][1]: r.append(i); c.append(ti[t])
TM=sp.csr_matrix((np.ones(len(r)),(r,c)),shape=(N,len(tags)))
def sub(th): cols=np.array([ti[t] for t in tags if t in auc and auc[t]>=th]); return TfidfTransformer().fit_transform(TM[:,cols]), len(cols)
for th in [0.85,0.75,0.0]:
    M,nt=sub(th)
    if th==0.75: ev(looKnn(M,30),f"visual tags AUC>={th} ({nt}) kNN k=30")
    # ridge: fonts with identical visual-tag vector get the same predicted doc -> linear map tags->doc, 5-fold
    from sklearn.linear_model import Ridge
    P=sp.lil_matrix(D.shape) if False else None
    preds=np.zeros((N,D.shape[1]),np.float32); fold=np.arange(N)%5
    Dd=D.tocsc()
    from sklearn.decomposition import TruncatedSVD
    svd=TruncatedSVD(256,random_state=0).fit(D); Z=svd.transform(D)
    Zp=np.zeros_like(Z)
    for f in range(5):
        tr=fold!=f; Zp[~tr]=Ridge(alpha=1.0).fit(M[tr],Z[tr]).predict(M[~tr])
    Hz=normalize(svd.transform(H)); Gz=normalize(Zp); rr=np.empty(N,int)
    for s0 in range(0,N,2000):
        S=Hz[s0:s0+2000]@Gz.T; t=S[np.arange(S.shape[0]),np.arange(s0,s0+S.shape[0])]; rr[s0:s0+S.shape[0]]=(S>t[:,None]).sum(1)
    print(f"  tags AUC>={th} ({nt}) ridge->LSA256 doc (5-fold)  R@1={np.mean(rr<1):.3f} R@10={np.mean(rr<10):.3f} R@100={np.mean(rr<100):.3f} medRank={np.median(rr)+1:.0f}/{N}",flush=True)
