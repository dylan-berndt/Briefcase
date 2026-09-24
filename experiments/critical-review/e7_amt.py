import numpy as np, pickle, csv, sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
lab,_=pickle.load(open("labels.pkl","rb"))
rows=list(csv.reader(open("dataset/AMT-testset/data.csv")))[1:]
rows=[[x.strip() for x in r] for r in rows]
trainset=[l.strip() for l in open("dataset/fontset/trainset") if l.strip()]
amtFonts=set(f for r in rows for f in r[1:4]); tagsAMT=sorted(set(r[0] for r in rows))
print("AMT triplets",len(rows),"tags",len(tagsAMT),"fonts",len(amtFonts),"AMT fonts in trainset:",len(amtFonts&set(trainset)))
def acc(score):  # score(tag,font)->float
    rng=np.random.RandomState(0); c=0
    for r in rows:
        s=np.array([score(r[0],f) for f in r[1:4]])+rng.rand(3)*1e-6
        c+=int(np.argmax(s)==int(r[4]))
    return c/len(rows)
print(f"chance: 0.333")
print(f"label oracle (font's own MyFonts tag list contains tag): {acc(lambda t,f: float(t in lab[f][1]) if f in lab else 0):.3f}")
def reps():
    keys=pickle.load(open("cache/all_keys.pkl","rb")); X=np.load("cache/all_X.npy")
    R={"ViT all.json":dict(zip(keys,X))}
    try: R["CLIP B/16 specimen"]=pickle.load(open("clip_myfonts.pkl","rb"))
    except Exception: pass
    for extra in sys.argv[1:]:
        R[extra]=pickle.load(open(extra,"rb"))
    return R
for name,E in reps().items():
    tr=[f for f in trainset if f in E and f in lab and f not in amtFonts]
    Xtr=np.stack([E[f] for f in tr]).astype(np.float64); mu=Xtr.mean(0); Xtr=normalize(Xtr-mu)
    probs={}
    for t in tagsAMT:
        y=np.array([t in lab[f][1] for f in tr])
        if y.sum()<5: continue
        clf=LogisticRegression(max_iter=2000,C=1.0,class_weight="balanced").fit(Xtr,y)
        for f in amtFonts:
            if f in E: probs[(t,f)]=clf.decision_function(normalize((E[f]-mu)[None]))[0]
    a=acc(lambda t,f: probs.get((t,f),-99))
    print(f"{name:25s} linear tag-probe (trained on {len(tr)} trainset fonts' MyFonts tags): AMT acc={a:.3f}")
