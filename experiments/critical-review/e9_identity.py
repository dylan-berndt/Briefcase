import pickle, numpy as np, sys
G=pickle.load(open("glyphemb.pkl","rb"))
keys=pickle.load(open("cache/all_keys.pkl","rb")); X=np.load("cache/all_X.npy"); ki={k:i for i,k in enumerate(keys)}
ks=[k for k in G]; print("fonts embedded:",len(ks))
E=np.stack([G[k][:26].astype(np.float32) for k in ks])  # [F,26,D] lowercase, normalized per glyph
def nv(v): return v/np.linalg.norm(v,axis=-1,keepdims=True)
rng=np.random.RandomState(0)
def rankTest(A,B,name,center=True):
    A=A.copy();B=B.copy()
    if center: mu=np.concatenate([A,B]).mean(0); A-=mu; B-=mu
    A=nv(A);B=nv(B); r=np.empty(len(A),int)
    for s in range(0,len(A),2000):
        S=A[s:s+2000]@B.T; t=S[np.arange(S.shape[0]),np.arange(s,s+S.shape[0])]; r[s:s+S.shape[0]]=(S>t[:,None]).sum(1)
    print(f"  {name:60s} R@1={np.mean(r<1):.3f} R@10={np.mean(r<10):.3f} medRank={np.median(r)+1:.0f}/{len(A)}")
print("Visual identity test: query = font's embedding from one glyph subset, gallery = all fonts' embedding from a DISJOINT glyph subset")
perm=rng.permutation(26); a,b=perm[:13],perm[13:]
rankTest(E[:,a].mean(1),E[:,b].mean(1),"13 lowercase letters vs other 13 (ViT, centered)")
rankTest(E[:,a].mean(1),E[:,b].mean(1),"13 vs 13 (ViT, uncentered raw cosine)",center=False)
rankTest(E[:,a[:3]].mean(1),E[:,b].mean(1),"3 letters vs other 13")
rankTest(E[:,a[:1]].mean(1),E[:,b].mean(1),"1 letter vs other 13")
# case
up=[i for i,k in enumerate(ks) if G[k].shape[0]==52]
Eu=np.stack([G[ks[i]][26:].astype(np.float32) for i in up]); El=E[up]
print(f"\nCase test on {len(up)} fonts with both cases:")
rankTest(El.mean(1),Eu.mean(1),"lowercase-26 vs UPPERCASE-26 of same font")
rankTest(El[:,a].mean(1),El[:,b].mean(1),"(reference: lowercase-13 vs lowercase-13, same fonts)")
# all.json vs lower-only
A=np.stack([X[ki[k]] for k in ks]); Lw=E.mean(1)
c=(nv(A)*nv(Lw)).sum(1); print(f"\ncos(all.json vector, true lowercase-26 vector): median={np.median(c):.4f} 10th pct={np.percentile(c,10):.4f}; frac <0.99: {np.mean(c<0.99):.3f}")
rankTest(Lw,A,"lowercase-26 vector vs all.json (case-mixed) vector of same font")
