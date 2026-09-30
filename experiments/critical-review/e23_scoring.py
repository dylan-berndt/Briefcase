# Font-scoring comparison for tag search, parser held fixed (oracle query tags from the ICCV query sets).
#   raw      : dot(query one-hot, sigmoid probs)          -- current TagSearch
#   tagZ     : dot(query, per-tag z-scored logits)        -- normalize each tag across fonts
#   semMulti : Turnbull et al. 2008 semantic multinomial: per-font probs normalized to sum 1 over the
#              vocabulary; multi-word query -> uniform query multinomial; rank by -KL(query || font)
#   both     : semantic multinomial computed from per-tag-normalized probabilities
import numpy as np, pickle, torch
from collections import Counter
from sklearn.metrics import average_precision_score
torch.manual_seed(0)
lab,_=pickle.load(open("labels.pkl","rb"))
keys=pickle.load(open("cache/all_keys.pkl","rb")); X=np.load("cache/all_X.npy"); E=dict(zip(keys,X))
rd=lambda f:[l.strip() for l in open(f) if l.strip()]
D="dataset/"
tr=[k for k in rd(D+"fontset/trainset") if k in E]; te=[k for k in rd(D+"fontset/testset") if k in E]
vocab=[t for t,_ in Counter(t for k in tr for t in lab[k][1]).most_common()]; vi={t:i for i,t in enumerate(vocab)}; V=len(vocab)
Y=np.zeros((len(tr),V),np.float32)
for i,k in enumerate(tr):
    for t in lab[k][1]: Y[i,vi[t]]=1
A=np.stack([E[k] for k in tr]); mu=A.mean(0); sd=A.std(0)+1e-6; f=lambda M:torch.tensor(((M-mu)/sd).astype(np.float32))
net=torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(512,1024),torch.nn.ReLU(),torch.nn.Dropout(0.3),torch.nn.Linear(1024,V))
opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-2); Xt=f(A); Yt=torch.tensor(Y)
for ep in range(40):
    net.train(); p=torch.randperm(len(Xt))
    for s in range(0,len(Xt),256):
        b=p[s:s+256]; l=torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]),Yt[b]); opt.zero_grad(); l.backward(); opt.step()
net.eval()
with torch.no_grad(): L=net(f(np.stack([E[k] for k in te]))).numpy()
P=1/(1+np.exp(-L))
Z=(L-L.mean(0))/(L.std(0)+1e-6)
semP=P/P.sum(1,keepdims=True)
Pz=1/(1+np.exp(-Z)); semZ=Pz/Pz.sum(1,keepdims=True)
G=[lab[k][1] for k in te]
def score(method,q):
    idx=[vi[t] for t in q]
    if method=="raw": return P[:,idx].sum(1)
    if method=="tagZ": return Z[:,idx].sum(1)
    S=semP if method=="semMulti" else semZ
    # -KL(q||s) with uniform q over query words = mean log s_w + const
    return np.log(S[:,idx]+1e-12).mean(1)
single=[[t] for t in rd(D+"myfonts-testset/singletag-test")]; multi=[q.split("&&&") for q in rd(D+"myfonts-testset/multitag-test")]
def distinct(method,qs):  # how different are rankings across queries? mean top-10 overlap between random query pairs
    rng=np.random.RandomState(0); tops=[set(np.argsort(-score(method,q))[:10]) for q in qs]
    pairs=rng.randint(len(qs),size=(500,2)); return np.mean([len(tops[a]&tops[b])/10 for a,b in pairs if a!=b])
for name,qs in [("single-tag (full)",single),("multi-tag",multi)]:
    qs=[q for q in qs if all(t in vi for t in q)]
    print(f"{name}: {len(qs)} queries, gallery {len(te)} test fonts")
    for m in ["raw","tagZ","semMulti","both"]:
        aps=[];p10=[]
        for q in qs:
            rel=np.array([all(t in g for t in q) for g in G],float)
            if rel.sum()==0: continue
            s=score(m,q); aps.append(average_precision_score(rel,s)); p10.append(rel[np.argsort(-s)[:10]].mean())
        print(f"  {m:9s} mAP={100*np.mean(aps):5.2f}  P@10={np.mean(p10):.3f}  top-10 overlap between random query pairs={distinct(m,qs):.3f}")
