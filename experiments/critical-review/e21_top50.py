# Music-tagging protocol (Won et al. 2020): top-50 most frequent tags, macro ROC-AUC and PR-AUC
# (= per-tag average precision) on the held-out test split. Applied to MyFonts with the frozen ViT.
import numpy as np, pickle, torch
from collections import Counter
from sklearn.metrics import roc_auc_score, average_precision_score
torch.manual_seed(0)
lab,_=pickle.load(open("labels.pkl","rb"))
keys=pickle.load(open("cache/all_keys.pkl","rb")); X=np.load("cache/all_X.npy"); E=dict(zip(keys,X))
rd=lambda f:[l.strip() for l in open(f) if l.strip()]
tr=[k for k in rd("dataset/fontset/trainset") if k in E]; te=[k for k in rd("dataset/fontset/testset") if k in E]
top=[t for t,_ in Counter(t for k in tr for t in lab[k][1]).most_common(50)]
def Y(ks): return np.array([[t in lab[k][1] for t in top] for k in ks],np.float32)
Ytr,Yte=Y(tr),Y(te)
A=np.stack([E[k] for k in tr]); mu=A.mean(0); sd=A.std(0)+1e-6; f=lambda M:torch.tensor(((M-mu)/sd).astype(np.float32))
net=torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(512,1024),torch.nn.ReLU(),torch.nn.Dropout(0.3),torch.nn.Linear(1024,50))
opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-2); Xt=f(A); Yt=torch.tensor(Ytr)
for ep in range(40):
    net.train(); p=torch.randperm(len(Xt))
    for s in range(0,len(Xt),256):
        b=p[s:s+256]; l=torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]),Yt[b]); opt.zero_grad(); l.backward(); opt.step()
net.eval()
with torch.no_grad(): P=net(f(np.stack([E[k] for k in te]))).numpy()
roc=[roc_auc_score(Yte[:,j],P[:,j]) for j in range(50)]; pr=[average_precision_score(Yte[:,j],P[:,j]) for j in range(50)]
prev=Yte.mean(0)
print(f"MyFonts top-50 tags, test fonts={len(te)}: macro ROC-AUC={np.mean(roc):.4f}  macro PR-AUC={np.mean(pr):.4f}  (random PR-AUC = mean tag prevalence = {prev.mean():.4f})")
print("tags:",", ".join(top))
order=np.argsort(roc)
print("worst 8:",[(top[j],round(roc[j],3)) for j in order[:8]]); print("best 8:",[(top[j],round(roc[j],3)) for j in order[-8:]])
