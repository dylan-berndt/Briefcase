import json, pickle, numpy as np, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
lab,_=pickle.load(open("labels.pkl","rb"))
qf=sys.argv[1]
Q=json.load(open("results/"+qf))
rng=np.random.RandomState(0)
keys=[k for k in Q if isinstance(Q[k],list) and len(Q[k])>=2]
hold=[];gal=[]
for k in keys:
    qs=list(Q[k]); i=rng.randint(len(qs)); hold.append(qs[i]); gal.append(" . ".join(qs[:i]+qs[i+1:]))
vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True).fit(gal)
G=normalize(vec.transform(gal)); H=normalize(vec.transform(hold))
src=np.array([lab[k][0] if k in lab else "?" for k in keys])
ranks=np.empty(len(keys),int)
for s in range(0,len(keys),2000):
    S=(H[s:s+2000]@G.T).toarray()
    t=S[np.arange(S.shape[0]),np.arange(s,s+S.shape[0])]
    ranks[s:s+S.shape[0]]=(S>t[:,None]).sum(1)
print(f"{qf}: TF-IDF text->text, one held-out query vs font's other queries, N={len(keys)} fonts")
for s in ["myfonts","dafont","google","ALL"]:
    m=(src==s) if s!="ALL" else np.ones(len(keys),bool)
    r=ranks[m]; print(f"  {s:8s} n={m.sum():6d} R@1={np.mean(r<1):.3f} R@10={np.mean(r<10):.3f} R@100={np.mean(r<100):.3f} median rank={np.median(r)+1:.0f} ({(np.median(r)+1)/len(keys)*100:.2f}%)")
np.save(f"ranks_tfidf_{qf}.npy",ranks)
