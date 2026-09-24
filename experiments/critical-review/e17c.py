exec(open("e17_fullscale.py").read().split('print(f"Full-scale')[0])
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(256,random_state=0).fit(D); Z=svd.transform(D); Hz=normalize(svd.transform(H)); fold=np.arange(N)%5
def ridgeEval(M,name,alpha=1.0):
    Zp=np.zeros_like(Z)
    for f in range(5):
        tr=fold!=f; Zp[~tr]=Ridge(alpha=alpha).fit(M[tr],Z[tr]).predict(M[~tr])
    Gz=normalize(Zp); rr=np.empty(N,int)
    for s0 in range(0,N,2000):
        S=Hz[s0:s0+2000]@Gz.T; t=S[np.arange(S.shape[0]),np.arange(s0,s0+S.shape[0])]; rr[s0:s0+S.shape[0]]=(S>t[:,None]).sum(1)
    print(f"  {name:48s} R@1={np.mean(rr<1):.3f} R@10={np.mean(rr<10):.3f} R@100={np.mean(rr<100):.3f} medRank={np.median(rr)+1:.0f}/{N}",flush=True)
ridgeEval(Z,"text->text in LSA256 (own other captions)")
V=np.stack([X[ki[k]] for k in F]); V=(V-V.mean(0))/V.std(0)
for a in [10,100,1000]: ridgeEval(V,f"ViT all.json ridge->doc alpha={a}",a)
# nonlinear: MLP ViT -> LSA doc, 5 fold
import torch
torch.manual_seed(0); Zp=np.zeros_like(Z); Vt=torch.tensor(V,dtype=torch.float32); Zt=torch.tensor(Z,dtype=torch.float32)
for f in range(5):
    tr=torch.tensor(fold!=f)
    net=torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(512,1024),torch.nn.ReLU(),torch.nn.Dropout(0.3),torch.nn.Linear(1024,256))
    opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-2); Xi=Vt[tr]; Yi=Zt[tr]
    for ep in range(40):
        net.train(); p=torch.randperm(len(Xi))
        for s in range(0,len(Xi),256):
            b=p[s:s+256]; loss=1-torch.nn.functional.cosine_similarity(net(Xi[b]),Yi[b]).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): Zp[~(fold!=f)]=net(Vt[~tr]).numpy()
Gz=normalize(Zp); rr=np.empty(N,int)
for s0 in range(0,N,2000):
    S=Hz[s0:s0+2000]@Gz.T; t=S[np.arange(S.shape[0]),np.arange(s0,s0+S.shape[0])]; rr[s0:s0+S.shape[0]]=(S>t[:,None]).sum(1)
print(f"  {'ViT all.json MLP->doc (cosine loss)':48s} R@1={np.mean(rr<1):.3f} R@10={np.mean(rr<10):.3f} R@100={np.mean(rr<100):.3f} medRank={np.median(rr)+1:.0f}/{N}")
