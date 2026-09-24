# Tag-bottleneck search engine: query --(learned from MyFonts captions)--> tag weights,
# font --(visual tagger trained on MyFonts)--> tag scores, rank by weighted sum.
# Everything is trained on MyFonts trainset only; evaluated on MyFonts testset and on DaFont.
import json, pickle, re, os, numpy as np, torch, pandas as pd, scipy.sparse as sp
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import average_precision_score
torch.manual_seed(0); rng=np.random.RandomState(0)
lab,_=pickle.load(open("labels.pkl","rb")); auc=pickle.load(open("tagauc_vit.pkl","rb"))
keys=pickle.load(open("cache_all_keys.pkl","rb")) if os.path.exists("cache_all_keys.pkl") else pickle.load(open("cache/all_keys.pkl","rb"))
X=np.load("cache/all_X.npy"); E=dict(zip(keys,X))
rd=lambda f:[l.strip() for l in open(f) if l.strip()]
D="dataset/"
tr=[k for k in rd(D+"fontset/trainset") if k in E]; va=[k for k in rd(D+"fontset/valset") if k in E]; te=[k for k in rd(D+"fontset/testset") if k in E]
Q=json.load(open("results/fontQueries.json"))
cnt=Counter(t for k in tr for t in lab[k][1]); vocab=[t for t,c in cnt.items() if c>=20]; vi={t:i for i,t in enumerate(vocab)}; V=len(vocab)
def Y(ks):
    y=np.zeros((len(ks),V),np.float32)
    for i,k in enumerate(ks):
        for t in lab[k][1]:
            if t in vi: y[i,vi[t]]=1
    return y
prior=Y(tr).mean(0)
# ---- visual tagger (font vector -> tag logits) ----
Xtr=np.stack([E[k] for k in tr]); mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; f=lambda A:torch.tensor(((A-mu)/sd).astype(np.float32))
net=torch.nn.Sequential(torch.nn.Dropout(0.2),torch.nn.Linear(512,1024),torch.nn.ReLU(),torch.nn.Dropout(0.3),torch.nn.Linear(1024,V))
opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-2); Xt=f(Xtr); Yt=torch.tensor(Y(tr))
for ep in range(40):
    net.train(); p=torch.randperm(len(Xt))
    for s in range(0,len(Xt),256):
        b=p[s:s+256]; l=torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]),Yt[b]); opt.zero_grad(); l.backward(); opt.step()
net.eval()
def tagScores(Vecs):
    with torch.no_grad(): return net(f(Vecs)).numpy()
# ---- learned query->tag model, trained on MyFonts TRAIN fonts' captions ----
capT=[];capY=[]
for k in tr:
    for q in Q.get(k,[]): capT.append(q); capY.append(k)
vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=30000,sublinear_tf=True).fit(capT)
C=normalize(vec.transform(capT)); CY=sp.csr_matrix(Y(capY))
def qtagKNN(texts,k=50):
    S=(normalize(vec.transform(texts))@C.T).toarray(); idx=np.argpartition(-S,k,1)[:,:k]
    W=np.zeros_like(S); r=np.arange(len(S))[:,None]; W[r,idx]=S[r,idx]; W/=W.sum(1,keepdims=True)+1e-9
    return (sp.csr_matrix(W)@CY).toarray()
qnet=torch.nn.Linear(len(vec.vocabulary_),V); qo=torch.optim.AdamW(qnet.parameters(),3e-3,weight_decay=1e-5)
Ccoo=C.tocoo(); Ct=torch.sparse_coo_tensor(np.vstack([Ccoo.row,Ccoo.col]),Ccoo.data.astype(np.float32),C.shape).coalesce()
CYd=torch.tensor(CY.toarray())
for ep in range(8):
    p=np.random.permutation(C.shape[0])
    for s in range(0,len(p),1024):
        b=p[s:s+1024]; xb=torch.tensor(C[b].toarray().astype(np.float32)); l=torch.nn.functional.binary_cross_entropy_with_logits(qnet(xb),CYd[b]); qo.zero_grad(); l.backward(); qo.step()
def qtagLinear(texts):
    with torch.no_grad(): return torch.sigmoid(qnet(torch.tensor(normalize(vec.transform(texts)).toarray().astype(np.float32)))).numpy()
# ---- lexical baseline (approximates TagSearch: phrase/lemma match of query words to tag names, log1p(idf) weights) ----
idf=np.log(len(tr)/np.maximum(Y(tr).sum(0),1)); tagRe=[re.compile(r"\b"+re.escape(t.replace("-"," "))+r"s?\b") for t in vocab]
def qtagLexical(texts):
    out=np.zeros((len(texts),V),np.float32)
    for i,q in enumerate(texts):
        ql=q.lower().replace("-"," ")
        for j,r in enumerate(tagRe):
            if vocab[j] in ("font","fonts","typeface") : continue
            if r.search(ql): out[i,j]=np.log1p(idf[j])
    return out
def weights(P,mode):
    if mode=="lexical": return P
    return np.maximum(P-prior,0)   # lift over the tag's base rate
def combo(texts):
    a=qtagLexical(texts); b=np.maximum(qtagKNN(texts)-prior,0)
    na=np.linalg.norm(a,axis=1,keepdims=True); nb=np.linalg.norm(b,axis=1,keepdims=True)
    return a/np.where(na>0,na,1)+b/np.where(nb>0,nb,1)
def rank(Wq,Z,visOnly):
    w=Wq*np.array([auc.get(t,0.5)>=0.75 for t in vocab]) if visOnly else Wq
    return w@Z.T
def zscore(S): return (S-S.mean(0))/(S.std(0)+1e-6)
# ======== Eval A: ICCV MyFonts-test queries, as TEXT through each query parser ========
Zte=zscore(tagScores(np.stack([E[k] for k in te]))); Gte=[lab[k][1] for k in te]
single=[[t] for t in rd(D+"myfonts-testset/singletag-test")]; multi=[q.split("&&&") for q in rd(D+"myfonts-testset/multitag-test")]; multi=[multi[i] for i in np.random.RandomState(1).choice(len(multi),1000,False)]
def evalTagQueries(qs,name):
    qs=[q for q in qs if all(t in vi for t in q)]
    texts=[" ".join(t.replace("-"," ") for t in q)+" font" for q in qs]
    rel=np.array([[all(t in g for t in q) for g in Gte] for q in qs],float); keep=rel.sum(1)>0; qs=[q for q,k in zip(qs,keep) if k]; texts=[t for t,k in zip(texts,keep) if k]; rel=rel[keep]
    oracle=np.zeros((len(qs),V)); 
    for i,q in enumerate(qs):
        for t in q: oracle[i,vi[t]]=1
    res={}
    for mname,W in [("oracle tags (ICCV-style)",oracle),("lexical match",weights(qtagLexical(texts),"lexical")),("learned kNN",weights(qtagKNN(texts),"x")),("learned linear",weights(qtagLinear(texts),"x")),("lexical+learned kNN",combo(texts))]:
        S=W@Zte.T; aps=[average_precision_score(rel[i],S[i]) for i in range(len(qs)) if S[i].std()>0]
        cov=np.mean([S[i].std()>0 for i in range(len(qs))])
        p10=np.mean([rel[i][np.argsort(-S[i])[:10]].mean() for i in range(len(qs))])
        print(f"  {name:10s} {mname:26s} mAP={100*np.mean(aps):5.1f} (queries with any match={cov:.2f})  P@10={p10:.3f}  [random P@10={rel.mean():.3f}]")
print(f"vocab={V} tags; tagger/text model trained on {len(tr)} MyFonts train fonts ({len(capT)} captions)")
print("== A. ICCV MyFonts-test tag queries phrased as text, gallery = MyFonts test fonts ==")
evalTagQueries(multi,"multi")
# ======== Eval B: real held-out captions of test fonts, attribute relevance of top-10 ========
visTags=[t for t in vocab if auc.get(t,0)>=0.85]
def vis(k): return set(t for t in lab[k][1] if t in visTags)
tq=[(k,Q[k][rng.randint(len(Q[k]))]) for k in te if k in Q and vis(k)]
texts=[q for _,q in tq]; tgt=[te.index(k) for k,_ in tq]
jac=np.array([[len(vis(a)&vis(b))/max(1,len(vis(a)|vis(b))) for b in te] for a,_ in tq])
print(f"== B. held-out natural-language captions ({len(tq)} test fonts), gallery = {len(te)} MyFonts test fonts ==")
for mname,W in [("lexical match",weights(qtagLexical(texts),"lexical")),("learned kNN",weights(qtagKNN(texts),"x")),("learned linear",weights(qtagLinear(texts),"x")),("lexical+learned kNN",combo(texts))]:
    S=W@Zte.T; top=np.argsort(-S,1)[:,:10]
    r=np.array([(S[i]>S[i,tgt[i]]).sum() for i in range(len(tq))])
    print(f"  {mname:16s} visual-tag Jaccard of top-10 vs target={np.mean([jac[i,top[i]].mean() for i in range(len(tq))]):.3f} [random {jac.mean():.3f}]  exact-font medRank={np.median(r)+1:.0f}/{len(te)} R@10={np.mean(r<10):.3f}")
pickle.dump(dict(vocab=vocab,prior=prior),open("e19_state.pkl","wb"))
# ======== Eval C: DaFont generalization (its own theme/category labels as relevance), both renderings ========
ok,A,B=pickle.load(open("pipeline_pairs.pkl","rb")); fm=pickle.load(open("fontname_map.pkl","rb"))
df=pd.read_csv("dafont/info.csv",on_bad_lines="skip"); f2={os.path.basename(str(r.filename)).lower():(r.category,r.theme) for r in df.itertuples()}
cats=[f2.get(os.path.basename(fm[k][0][1]).lower(),("?","?")) for k in ok]
labels=[(c,0) for c,n in Counter(x[0] for x in cats).items() if n>=10 and c not in("?",)]+[(t,1) for t,n in Counter(x[1] for x in cats).items() if n>=8 and t not in ("?","Various")]
texts=[l.replace("_"," ").lower()+" font" for l,_ in labels]; rel=np.array([[c[j]==l for c in cats] for l,j in labels],float)
print(f"== C. DaFont ({len(ok)} fonts), queries = its own {len(labels)} category/theme names as text, relevance = that label ==")
for rname,Vv in [("metric render (as in all.json)",A.mean(1)),("MyFonts-style render",B.mean(1))]:
    Z=zscore(tagScores(Vv))
    for mname,W in [("lexical match",weights(qtagLexical(texts),"lexical")),("learned kNN",weights(qtagKNN(texts),"x")),("learned linear",weights(qtagLinear(texts),"x")),("lexical+learned kNN",combo(texts))]:
        S=W@Z.T; aps=[average_precision_score(rel[i],S[i]) for i in range(len(labels))]; p10=np.mean([rel[i][np.argsort(-S[i])[:10]].mean() for i in range(len(labels))])
        print(f"  {rname:30s} {mname:16s} mAP={100*np.mean(aps):5.1f}  P@10={p10:.3f}  [random mAP~{100*rel.mean(1).mean():.1f}]")
