import os, json, pandas as pd, pickle, glob, re
from collections import defaultdict
lab={}  # queryKey -> (source, frozenset(labels))
# myfonts
for f in os.listdir("dataset/taglabel"):
    lab[f]=("myfonts",frozenset(open("dataset/taglabel/"+f).read().split()))
# dafont
df=pd.read_csv("dafont/info.csv",on_bad_lines="skip")
for n,g in df.groupby("base_font_name"):
    lab.setdefault(n,("dafont",frozenset(str(x) for x in set(g.category)|set(g.theme))))
# google: tags + category from METADATA
gt=pd.read_csv("google/fonts/tags/all/families.csv",names=["family","na","tags","weight"])
gtags=defaultdict(dict)
for r in gt.itertuples(): gtags[r.family][r.tags.split("/")[-1]]=r.weight
for md in glob.glob("google/fonts/*/*/METADATA.pb"):
    t=open(md).read(); name=re.search(r'^name: "(.*)"',t,re.M); cats=re.findall(r'^category: "(.*)"',t,re.M)
    if not name: continue
    n=name.group(1)
    lab.setdefault(n,("google",frozenset(list(gtags.get(n,{}).keys())+cats)))
pickle.dump((lab,dict(gtags)),open("labels.pkl","wb"))
from collections import Counter
print(Counter(s for s,_ in lab.values()))
