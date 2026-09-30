import numpy as np, pickle, sys, torch
from collections import Counter
lab,_=pickle.load(open("labels.pkl","rb"))
def load(path):
    if path=="all.json":
        keys=pickle.load(open("cache/all_keys.pkl","rb")); X=np.load("cache/all_X.npy"); return dict(zip(keys,X))
    return pickle.load(open(path,"rb"))
E=load(sys.argv[1])
rd=lambda f:[l.strip() for l in open(f) if l.strip()]
tr=[k for k in rd("dataset/fontset/trainset") if k in E]; te=[k for k in rd("dataset/fontset/testset") if k in E]
if len(sys.argv)>2:
    common=set(pickle.load(open(sys.argv[2],"rb"))); tr=[k for k in tr if k in common]
single=rd("dataset/myfonts-testset/singletag-test"); multi=[q.split("&&&") for q in rd("dataset/myfonts-testset/multitag-test")]
vocab=sorted(set(t for k in tr for t in lab[k][1])|set(single)|set(t for q in multi for t in q)); vi={t:i for i,t in enumerate(vocab)}
Y=np.zeros((len(tr),len(vocab)),np.float32)
for i,k in enumerate(tr):
    for t in lab[k][1]: Y[i,vi[t]]=1
Xtr=np.stack([E[k] for k in tr]).astype(np.float32); mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
f=lambda A:(A-mu)/sd
Xtr=torch.tensor(f(Xtr)); Xte=torch.tensor(f(np.stack([E[k] for k in te]).astype(np.float32))); Yt=torch.tensor(Y)
torch.manual_seed(0)
net=torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(Xtr.shape[1],1024),torch.nn.ReLU(),torch.nn.Dropout(0.3),torch.nn.Linear(1024,len(vocab)))
opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-2)
for ep in range(40):
    net.train(); perm=torch.randperm(len(Xtr))
    for s in range(0,len(Xtr),256):
        b=perm[s:s+256]; loss=torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr[b]),Yt[b]); opt.zero_grad(); loss.backward(); opt.step()
net.eval()
with torch.no_grad(): L=torch.nn.functional.logsigmoid(net(Xte)).numpy()
G=[lab[k][1] for k in te]
def ap(scores,rel):
    o=np.argsort(-scores); r=rel[o]
    if r.sum()==0: return None
    h=np.cumsum(r); return float((h[r==1]/(np.where(r==1)[0]+1)).mean())
def ndcg(scores,rel):
    o=np.argsort(-scores); r=rel[o]; d=1/np.log2(np.arange(2,len(r)+2)); ideal=np.sort(rel)[::-1]
    return float((r*d).sum()/(ideal*d).sum()) if rel.sum() else None
def run(queries):
    aps=[];nd=[]
    for q in queries:
        rel=np.array([all(t in g for t in q) for g in G],float); s=L[:,[vi[t] for t in q]].sum(1)
        a=ap(s,rel)
        if a is not None: aps.append(a); nd.append(ndcg(s,rel))
    return 100*np.mean(aps),100*np.mean(nd),len(aps)
cnt=Counter(t for k in tr for t in lab[k][1]); top300=set(t for t,_ in cnt.most_common(300))
s300=[[t] for t in single if t in top300]; sfull=[[t] for t in single]
print(f"{sys.argv[1]}: train={len(tr)} test={len(te)}")
for name,q in [("single tag (300)",s300),("single tag (full)",sfull),("multi tag",multi)]:
    a,n,c=run(q); print(f"  {name:18s} mAP={a:.2f} NDCG={n:.2f} (queries={c})")
