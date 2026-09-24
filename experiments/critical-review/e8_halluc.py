import json, re, glob, numpy as np
cat={}
for md in glob.glob("google/fonts/*/*/METADATA.pb"):
    t=open(md).read(); n=re.search(r'^name: "(.*)"',t,re.M)
    if n: cat[n.group(1)]=set(re.findall(r'^category: "(.*)"',t,re.M))
sans=re.compile(r"\bsans[- ]?serif|\bsans\b|grotesk|grotesque",re.I)
serif=re.compile(r"(?<!sans )(?<!sans-)(?<!sans)\bserif(?!s? ?-?less)|\bslab\b|old[- ]style|didone|transitional serif",re.I)
script=re.compile(r"\bscript\b|handwrit|calligraph|cursive|brush",re.I)
mono=re.compile(r"monospace|mono-spaced|monospaced|fixed[- ]width|typewriter|coding",re.I)
for qf in ["fontQueries.json","fontQueriesV2.json"]:
    Q=json.load(open("results/"+qf))
    stats={}
    for fam,qs in Q.items():
        if fam not in cat or not isinstance(qs,list): continue
        c=cat[fam]
        for q in qs:
            s=bool(sans.search(q)); se=bool(serif.search(re.sub(sans,"",q))); sc=bool(script.search(q)); mo=bool(mono.search(q))
            for C in c:
                d=stats.setdefault(C,[0,0,0,0,0]); d[0]+=1; d[1]+=s; d[2]+=se; d[3]+=sc; d[4]+=mo
    print(f"\n{qf}: fraction of generated queries mentioning each class, by TRUE Google category")
    print(f"  {'category':14s} {'nQueries':>8s} {'sans':>6s} {'serif':>6s} {'script':>6s} {'mono':>6s}")
    for C,(n,a,b,c_,d) in sorted(stats.items()):
        print(f"  {C:14s} {n:8d} {a/n:6.3f} {b/n:6.3f} {c_/n:6.3f} {d/n:6.3f}")
