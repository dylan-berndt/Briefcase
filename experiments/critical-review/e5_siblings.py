import json, pickle, numpy as np
from pygtrie import CharTrie
from collections import Counter, defaultdict
keys=pickle.load(open("cache/all_keys.pkl","rb")); src=pickle.load(open("all_sources.pkl","rb"))
lab,_=pickle.load(open("labels.pkl","rb"))
fm=pickle.load(open("fontname_map.pkl","rb"))
for qf in ["fontQueries.json","fontQueriesV2.json"]:
    Q=json.load(open("results/"+qf))
    t=CharTrie()
    for k in Q: t[k]=k
    fam=defaultdict(list)
    for k in keys:
        m=t.longest_prefix(k.strip())
        if m.key is not None: fam[m.key].append(k)
    sizes=np.array([len(v) for v in fam.values()])
    print(f"\n{qf}: {len(fam)} described families matched to {sizes.sum()} corpus entries")
    for s in ["myfonts","dafont","google"]:
        z=np.array([len(v) for f,v in fam.items() if lab.get(f,('?',))[0]==s])
        if len(z): print(f"  {s}: families={len(z)} entries/family mean={z.mean():.2f} frac families with >1 entry={np.mean(z>1):.3f} max={z.max()}  -> R@1 ceiling if siblings count as misses (uniform) ~{np.mean(1/z):.3f}")
    # suspicious prefix matches: matched key is a strict prefix not at word boundary, or the entry's own family name (from font file) differs
    bad=[]
    for f,v in fam.items():
        for k in v:
            rest=k.strip()[len(f):]
            if rest and not rest.startswith(" "): bad.append((f,k))
    print("  entries matched by non-word-boundary prefix:",len(bad), bad[:8])
    # dafont/google: entry family name from file != described family
    wrong=0; tot=0; ex=[]
    for f,v in fam.items():
        if lab.get(f,('?',))[0]!="dafont": continue
        for k in v:
            if "dafont" not in src.get(k,()): continue
            tot+=1
    unmatched=sum(1 for k in keys if t.longest_prefix(k.strip()).key is None)
    print("  corpus entries with no description at all (pure distractors):",unmatched, "of",len(keys))
