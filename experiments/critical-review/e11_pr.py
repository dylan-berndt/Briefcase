import numpy as np, pandas as pd, io, random, os
from PIL import Image
from render import rochesterFile
def pr(X):
    X=X-X.mean(0); ev=np.linalg.svd(X,compute_uv=False)**2
    return ev.sum()**2/(ev**2).sum(), np.searchsorted(np.cumsum(ev)/ev.sum(),0.9)+1
def prUncentered(X):
    ev=np.linalg.svd(X,compute_uv=False)**2; return ev.sum()**2/(ev**2).sum()
def knn1(X,y):
    sq=(X**2).sum(1); S=-(sq[:,None]+sq[None]-2*X@X.T)
    np.fill_diagonal(S,-np.inf); return np.mean(y[S.argmax(1)]==y)
df=pd.read_parquet("mnist.parquet"); rng=np.random.RandomState(0); idx=rng.choice(len(df),3000,False)
M=np.stack([np.array(Image.open(io.BytesIO(df.image.iloc[i]["bytes"])),np.float32)/255 for i in idx]).reshape(3000,-1); my=df.label.values[idx]
p,d90=pr(M); print(f"MNIST pixels (784-d, n=3000): centered PR={p:.1f} dims-for-90%var={d90} uncentered PR={prUncentered(M):.1f}  1-NN digit acc={knn1(M,my):.3f}")
keys=sorted(os.listdir("dataset/taglabel")); random.Random(0).shuffle(keys)
fonts=keys[:600]; L="aegnrs"
imgs=[];lab=[];fid=[]
for fi,k in enumerate(fonts):
    for li,c in enumerate(L):
        g=rochesterFile(f"dataset/fontimage/{k}_{c}.png")
        if g is not None: imgs.append(g.reshape(-1)); lab.append(li); fid.append(fi)
G=np.stack(imgs); lab=np.array(lab); fid=np.array(fid)
p,d90=pr(G); print(f"Font glyphs, mixed letters (2304-d, n={len(G)}): centered PR={p:.1f} dims90={d90} uncentered PR={prUncentered(G):.1f}  1-NN letter acc={knn1(G[:3000],lab[:3000]):.3f}")
for li,c in enumerate(L[:3]):
    p,d90=pr(G[lab==li]); print(f"  single letter '{c}' across {np.sum(lab==li)} fonts: centered PR={p:.1f} dims90={d90}")
