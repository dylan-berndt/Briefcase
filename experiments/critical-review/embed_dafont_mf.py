import pickle, numpy as np, time
from multiprocessing import Pool
from render import myfontsStyleFromFont
L="abcdefghijklmnopqrstuvwxyz"
keys=pickle.load(open("cache/all_keys.pkl","rb")); src=pickle.load(open("all_sources.pkl","rb")); fm=pickle.load(open("fontname_map.pkl","rb"))
D=[k for k in keys if src[k]=={"dafont"}]
def job(k):
    try: imgs=[myfontsStyleFromFont(fm[k][0][1],c) for c in L]
    except Exception: return k,None
    if any(i is None for i in imgs): return k,None
    return k,np.stack(imgs)
if __name__=="__main__":
    from model import *; torch.set_num_threads(3); m=load(); out={}; t=time.time()
    with Pool(1) as pool:
        for n,(k,im) in enumerate(pool.imap(job,D,chunksize=16)):
            if im is not None:
                E=embedGlyphs(m,list(im),bs=64); E/=np.linalg.norm(E,axis=1,keepdims=True); out[k]=E.mean(0).astype(np.float32)
            if n%1000==0: print(n,len(out),f"{time.time()-t:.0f}s",flush=True); pickle.dump(out,open("dafont_mf.pkl","wb"))
    pickle.dump(out,open("dafont_mf.pkl","wb")); print("done",len(out),flush=True)
