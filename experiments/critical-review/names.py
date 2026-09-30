import os, glob, pickle, sys
from PIL import ImageFont
from multiprocessing import Pool
def nm(p):
    try:
        f=ImageFont.truetype(p,32); a,b=f.getname(); return p, f"{a} {b}"
    except Exception as e: return p, None
if __name__=="__main__":
    out={}
    for src,root in [("google","google/fonts"),("dafont","dafont/fonts")]:
        ps=glob.glob(root+"/**/*.ttf",recursive=True)+glob.glob(root+"/**/*.otf",recursive=True)
        with Pool(4) as pool:
            for p,n in pool.imap_unordered(nm,ps,chunksize=64):
                if n: out.setdefault(n,[]).append((src,p))
        print(src,len(ps))
    pickle.dump(out,open("fontname_map.pkl","wb"))
    keys=pickle.load(open("cache/all_keys.pkl","rb"))
    tags=set(os.listdir("dataset/taglabel"))
    src={}
    for k in keys:
        s=set()
        if k in tags: s.add("myfonts")
        for a,_ in out.get(k,[]): s.add(a)
        src[k]=s
    from collections import Counter
    print(Counter(tuple(sorted(v)) for v in src.values()))
    pickle.dump(src,open("all_sources.pkl","wb"))
