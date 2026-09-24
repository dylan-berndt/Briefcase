import pickle, numpy as np, pandas as pd, torch, os
from sklearn.metrics import roc_auc_score
ok,A,B=pickle.load(open("pipeline_pairs.pkl","rb"))
fm=pickle.load(open("fontname_map.pkl","rb")); lab,_=pickle.load(open("labels.pkl","rb"))
def nv(v): return v/np.linalg.norm(v,axis=-1,keepdims=True)
def rank(Q,G,name):
    mu=np.concatenate([Q,G]).mean(0); Q=nv(Q-mu); G=nv(G-mu); S=Q@G.T; t=np.diag(S); r=(S>t[:,None]).sum(1)
    print(f"  {name:62s} R@1={np.mean(r<1):.3f} R@10={np.mean(r<10):.3f} medRank={np.median(r)+1:.0f}/{len(Q)}")
print(f"{len(ok)} DaFont fonts rendered through both pipelines (26 lowercase each)")
p=np.random.RandomState(0).permutation(26); a,b=p[:13],p[13:]
rank(A[:,a].mean(1),A[:,b].mean(1),"reference: metric 13 letters vs metric other 13")
rank(B[:,a].mean(1),A[:,b].mean(1),"MyFonts-style 13 letters vs metric other 13 (pipeline swap)")
rank(B.mean(1),A.mean(1),"MyFonts-style 26 vs metric 26 (same letters, pipeline swap)")
# tag transfer: MLP probe trained on MyFonts all.json (MyFonts pipeline)
exec(open("e13_iccv.py").read().split("net.eval()")[0].replace("E=load(sys.argv[1])","E=load('all.json')").replace("if len(sys.argv)>2:","if False:"))
net.eval()
df=pd.read_csv("dafont/info.csv",on_bad_lines="skip"); f2cat={}
for r in df.itertuples(): f2cat[os.path.basename(str(r.filename)).lower()]=(r.category,r.theme)
cats=[f2cat.get(os.path.basename(fm[k][0][1]).lower(),("?","?")) for k in ok]
def score(V,tag):
    with torch.no_grad(): return net(torch.tensor(f(V.astype(np.float32))))[:,vi[tag]].numpy()
print("DaFont category detection by MyFonts-trained tag probe (AUC), metric pipeline vs MyFonts-style pipeline:")
for cat,tag in [("Script","script"),("Gothic","blackletter"),("Techno","techno"),("Basic","sans-serif"),("Fancy","decorative")]:
    y=np.array([c[0]==cat for c in cats])
    if y.sum()<10: continue
    print(f"  {cat:8s}->{tag:12s} n+={y.sum():4d}  AUC metric={roc_auc_score(y,score(A.mean(1),tag)):.3f}  MyFonts-style={roc_auc_score(y,score(B.mean(1),tag)):.3f}")
for th,tag in [("Handwritten","handwrite"),("Serif","serif"),("Sans serif","sans-serif"),("Pixel, bitmap","pixel"),("Comic","comic"),("Medieval","medieval"),("Western","western"),("Stencil, army","stencil"),("Retro","retro"),("Calligraphy","calligraphy")]:
    y=np.array([c[1]==th for c in cats])
    if y.sum()<10: continue
    print(f"  theme {th:14s}->{tag:10s} n+={y.sum():4d}  AUC metric={roc_auc_score(y,score(A.mean(1),tag)):.3f}  MyFonts-style={roc_auc_score(y,score(B.mean(1),tag)):.3f}")
