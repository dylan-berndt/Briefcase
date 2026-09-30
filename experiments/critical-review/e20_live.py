# Run the tag-bottleneck search over the DaFont corpus for the 30 pre-existing queries in
# experiments/choice-searches/estimateSearchQuality.py, two ways:
#   A = current vectors (all.json, metric render, case-mixed)
#   B = MyFonts-style re-render (dafont_mf.pkl)
# Same MyFonts-trained tagger, same lexical parser. Saves top-12 per system per query + sample tiles.
import json, pickle, re, os, zlib, numpy as np, torch, random
from collections import Counter
from PIL import Image, ImageFont, ImageDraw
torch.manual_seed(0)
src_q=open("experiments/choice-searches/estimateSearchQuality.py").read()
QUERIES=eval(re.search(r"QUERIES = (\[.*?\])",src_q,re.S).group(1))
lab,_=pickle.load(open("labels.pkl","rb"))
keys=pickle.load(open("cache/all_keys.pkl","rb")); X=np.load("cache/all_X.npy"); E=dict(zip(keys,X))
src=pickle.load(open("all_sources.pkl","rb")); fm=pickle.load(open("fontname_map.pkl","rb"))
MF=pickle.load(open("dafont_mf.pkl","rb"))
rd=lambda f:[l.strip() for l in open(f) if l.strip()]
tr=[k for k in rd("dataset/fontset/trainset")+rd("dataset/fontset/valset") if k in E]
cnt=Counter(t for k in tr for t in lab[k][1]); vocab=[t for t,c in cnt.items() if c>=20]; vi={t:i for i,t in enumerate(vocab)}; V=len(vocab)
Y=np.zeros((len(tr),V),np.float32)
for i,k in enumerate(tr):
    for t in lab[k][1]:
        if t in vi: Y[i,vi[t]]=1
Xtr=np.stack([E[k] for k in tr]); mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; f=lambda A:torch.tensor(((A-mu)/sd).astype(np.float32))
net=torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(512,1024),torch.nn.ReLU(),torch.nn.Dropout(0.3),torch.nn.Linear(1024,V))
opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-2); Xt=f(Xtr); Yt=torch.tensor(Y)
for ep in range(40):
    net.train(); p=torch.randperm(len(Xt))
    for s in range(0,len(Xt),256):
        b=p[s:s+256]; l=torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]),Yt[b]); opt.zero_grad(); l.backward(); opt.step()
net.eval()
idf=np.log(len(tr)/np.maximum(Y.sum(0),1)); STOP={"font","fonts","typeface","typefaces"}
tagRe=[re.compile(r"\b"+re.escape(t.replace("-"," "))+r"s?\b") for t in vocab]
def parse(q):
    ql=q.lower().replace("-"," "); w=np.zeros(V,np.float32)
    for j,r in enumerate(tagRe):
        if vocab[j] not in STOP and r.search(ql): w[j]=np.log1p(idf[j])
    return w
gallery=[k for k in keys if src[k]=={"dafont"} and k in MF]
def Z(vecs):
    with torch.no_grad(): S=net(f(vecs)).numpy()
    return (S-S.mean(0))/(S.std(0)+1e-6)
ZA=Z(np.stack([E[k] for k in gallery])); ZB=Z(np.stack([MF[k] for k in gallery]))
os.makedirs("tiles",exist_ok=True)
def tile(k):
    fn=f"tiles/{zlib.crc32(k.encode())}.png"
    if os.path.exists(fn): return fn
    im=Image.new("L",(360,56),255); d=ImageDraw.Draw(im)
    try:
        font=ImageFont.truetype(fm[k][0][1],30); d.text((8,8),"Handgloves Quiz",font=font,fill=0)
    except Exception: d.text((8,20),"(render failed)",fill=0)
    im.save(fn,optimize=True); return fn
out=[]
for q in QUERIES:
    w=parse(q); matched=[vocab[i] for i in np.nonzero(w)[0]]
    rec={"query":q,"matchedTags":matched}
    for name,Zs in [("A",ZA),("B",ZB)]:
        top=np.argsort(-(Zs@w))[:8] if w.any() else []
        rec[name]=[{"key":gallery[i],"tile":tile(gallery[i])} for i in top]
    rec["R"]=[{"key":gallery[i],"tile":tile(gallery[i])} for i in np.random.RandomState(zlib.crc32(q.encode())).choice(len(gallery),8,False)]
    out.append(rec)
    print(f"{q!r}: tags={matched}")
json.dump({"gallery":len(gallery),"results":out},open("e20_results.json","w"),indent=1)
print("gallery",len(gallery))
