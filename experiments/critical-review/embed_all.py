import pickle, numpy as np, random, os, sys, time
from multiprocessing import Pool
from render import *
L="abcdefghijklmnopqrstuvwxyz"
keys=pickle.load(open("cache/all_keys.pkl","rb")); src=pickle.load(open("all_sources.pkl","rb")); fm=pickle.load(open("fontname_map.pkl","rb"))
keys=[k for k in keys if src[k]=={"myfonts"}]; rng=random.Random(0); upperSet=set(rng.sample(keys,3000))
def job(k):
    chars=list(L)+([c.upper() for c in L] if k in upperSet else [])
    try:
        if "myfonts" in src[k]:
            imgs=[rochesterFile(f"dataset/fontimage/{k}_{c if c.islower() else c*2}.png") for c in chars]
        elif src[k]:
            p=fm[k][0][1]; imgs=[metricRender(p,c) for c in chars]
        else: return k,None
    except Exception as e: return k,None
    if any(i is None for i in imgs): return k,None
    return k,np.stack(imgs).astype(np.float16)
if __name__=="__main__":
    from model import *; torch.set_num_threads(3)
    m=load(); out={}; imgs_out={}; t=time.time()
    with Pool(1) as pool:
        for n,(k,im) in enumerate(pool.imap(job,keys,chunksize=16)):
            if im is not None:
                E=embedGlyphs(m,list(im.astype(np.float32)),bs=64)
                out[k]=(E/np.linalg.norm(E,axis=1,keepdims=True)).astype(np.float16); imgs_out[k]=(im.astype(np.float32)*255).round().astype(np.uint8)
            if n%500==0:
                print(n,len(out),f"{time.time()-t:.0f}s",flush=True)
            if n%5000==0 and n: pickle.dump(out,open("glyphemb.pkl","wb")); pickle.dump(imgs_out,open("glyphimg.pkl","wb"))
    pickle.dump(out,open("glyphemb.pkl","wb")); pickle.dump(imgs_out,open("glyphimg.pkl","wb")); print("done",len(out))
