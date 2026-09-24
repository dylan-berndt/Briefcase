import json, pickle, numpy as np
from collections import Counter
lab,gtags=pickle.load(open("labels.pkl","rb"))
for qf in ["fontQueries.json","fontQueriesV2.json"]:
    Q=json.load(open("results/"+qf))
    ks=[k for k in Q if k in lab]
    print(f"\n== {qf}: {len(Q)} fonts, {len(ks)} with resolvable source labels", Counter(lab[k][0] for k in ks))
    groups=Counter((lab[k][0],lab[k][1]) for k in ks)
    for s in ["myfonts","dafont","google","ALL"]:
        sk=[k for k in ks if s=="ALL" or lab[k][0]==s]
        gs=np.array([groups[(lab[k][0],lab[k][1])] for k in sk])
        # labels identical across sources count too? keep per-source; ceiling if query conveys only labels:
        for K in [1,10]:
            pass
        ceil1=np.mean(np.minimum(1,1/gs)); ceil10=np.mean(np.minimum(1,10/gs))
        print(f"  {s:8s} n={len(sk):6d} unique-labelset frac={np.mean(gs==1):.3f} median group={np.median(gs):.0f} mean group={gs.mean():.1f} | label-only ceiling R@1={ceil1:.3f} R@10={ceil10:.3f}")
    nl=[len(lab[k][1]) for k in ks if lab[k][0]=="dafont"]; print("  dafont #labels per font:",Counter(nl).most_common(4))
    nl=[len(lab[k][1]) for k in ks if lab[k][0]=="myfonts"]; print("  myfonts #tags per font: median",np.median(nl), "pct<=3:",np.mean(np.array(nl)<=3))
