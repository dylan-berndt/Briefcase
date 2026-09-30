# Same probe as e13 (ICCV protocol), but also score it with retrieval.py@8b28fb1's meanAP:
# per-tag AP within random 256-item batches, averaged over tags with >=1 positive in the batch.
import numpy as np, pickle, torch
from sklearn.metrics import average_precision_score
exec(open("e13_iccv.py").read().split("G=[lab[k][1] for k in te]")[0].replace("E=load(sys.argv[1])","E=load('all.json')").replace("if len(sys.argv)>2:","if False:"))
Yte=np.zeros((len(te),len(vocab)))
for i,k in enumerate(te):
    for t in lab[k][1]: Yte[i,vi[t]]=1
P=L
def batchMAP(P,Y,bs=256,n=200,seed=0):
    r=np.random.RandomState(seed); out=[]
    for _ in range(n):
        b=r.choice(len(Y),bs,False); aps=[average_precision_score(Y[b,c],P[b,c]) for c in range(Y.shape[1]) if Y[b,c].sum()>0]
        out.append(np.mean(aps))
    return 100*np.mean(out)
def corpusMAP(P,Y):
    return 100*np.mean([average_precision_score(Y[:,c],P[:,c]) for c in range(Y.shape[1]) if Y[:,c].sum()>0])
rnd=np.random.RandomState(1).rand(*P.shape)
print(f"ViT probe  batch-mAP(256, retrieval.py style)={batchMAP(P,Yte):.1f}   corpus-mAP over {len(te)} test fonts={corpusMAP(P,Yte):.1f}")
print(f"RANDOM     batch-mAP(256, retrieval.py style)={batchMAP(rnd,Yte):.1f}   corpus-mAP={corpusMAP(rnd,Yte):.1f}")
