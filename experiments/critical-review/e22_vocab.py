# Tag frequency distribution and tagger ROC-AUC by tag frequency (MyFonts, official splits, frozen ViT + MLP).
import numpy as np, pickle, torch
from collections import Counter
from sklearn.metrics import roc_auc_score
torch.manual_seed(0)
lab,_=pickle.load(open("labels.pkl","rb"))
keys=pickle.load(open("cache/all_keys.pkl","rb")); X=np.load("cache/all_X.npy"); E=dict(zip(keys,X))
rd=lambda f:[l.strip() for l in open(f) if l.strip()]
tr=[k for k in rd("dataset/fontset/trainset") if k in E]; te=[k for k in rd("dataset/fontset/testset")+rd("dataset/fontset/valset") if k in E]
cnt=Counter(t for k in tr for t in lab[k][1])
allTags=Counter(t for k in lab if lab[k][0]=="myfonts" for t in lab[k][1])
print(f"distinct MyFonts tags (all fonts): {len(allTags)}; in train split: {len(cnt)}")
for th in [1,5,10,20,50,100,200,500]: print(f"  tags with >= {th:>3} train fonts: {sum(c>=th for c in cnt.values())}")
tpf=[len(lab[k][1]) for k in tr]; print(f"tags per font: median {np.median(tpf):.0f}, mean {np.mean(tpf):.1f}")
vocab=[t for t,c in cnt.most_common() if c>=20]; vi={t:i for i,t in enumerate(vocab)}
def Y(ks):
    y=np.zeros((len(ks),len(vocab)),np.float32)
    for i,k in enumerate(ks):
        for t in lab[k][1]:
            if t in vi: y[i,vi[t]]=1
    return y
A=np.stack([E[k] for k in tr]); mu=A.mean(0); sd=A.std(0)+1e-6; f=lambda M:torch.tensor(((M-mu)/sd).astype(np.float32))
net=torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(512,1024),torch.nn.ReLU(),torch.nn.Dropout(0.3),torch.nn.Linear(1024,len(vocab)))
opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-2); Xt=f(A); Yt=torch.tensor(Y(tr))
for ep in range(40):
    net.train(); p=torch.randperm(len(Xt))
    for s in range(0,len(Xt),256):
        b=p[s:s+256]; l=torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]),Yt[b]); opt.zero_grad(); l.backward(); opt.step()
net.eval()
with torch.no_grad(): P=net(f(np.stack([E[k] for k in te]))).numpy()
Yte=Y(te); auc={}
for t,j in vi.items():
    if Yte[:,j].sum()>=5: auc[t]=roc_auc_score(Yte[:,j],P[:,j])
print(f"\nROC-AUC by train frequency (test+val fonts={len(te)}, tags with >=5 held-out positives):")
for lo,hi in [(500,10**9),(200,500),(100,200),(50,100),(20,50)]:
    a=[auc[t] for t in auc if lo<=cnt[t]<hi]
    if a: print(f"  {lo:>4}-{hi if hi<10**9 else '':<4} train fonts: n={len(a):4d} tags  median AUC {np.median(a):.3f}  frac AUC>=0.75: {np.mean(np.array(a)>=0.75):.2f}")
print(f"tags with >=50 train fonts AND AUC>=0.75: {sum(1 for t in auc if cnt[t]>=50 and auc[t]>=0.75)}; AUC>=0.80: {sum(1 for t in auc if cnt[t]>=50 and auc[t]>=0.80)}")
pickle.dump((dict(cnt),auc),open("e22_vocab.pkl","wb"))
# near-duplicate spellings
import re
norm=lambda t:re.sub(r"[^a-z]","",t.lower())
groups={}
for t in cnt:
    if cnt[t]>=20: groups.setdefault(norm(t).rstrip("s"),[]).append(t)
dups=[g for g in groups.values() if len(g)>1]
print(f"\nspelling-variant groups among tags with >=20 fonts: {len(dups)}; e.g.", dups[:12])
for fam in [["sans-serif","sans","sanserif","sansserif"],["handwrite","handwritten","handwriting","hand","hand-drawn","handmade"],["script","cursive","calligraphy","calligraphic"],["retro","vintage","old-fashioned","antique"],["heavy","bold","black","fat"]]:
    print("  ",[(t,cnt.get(t,0),round(auc.get(t,float('nan')),3)) for t in fam])
