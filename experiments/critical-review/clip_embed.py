import torch, numpy as np, pickle, os, time
from transformers import CLIPModel, CLIPProcessor
from multiprocessing import Pool
from specimen import specimen
torch.set_num_threads(3)
MID="openai/clip-vit-base-patch16"
def job(k):
    try: im=specimen(k)
    except Exception as e: im=None
    return k,im
if __name__=="__main__":
    model=CLIPModel.from_pretrained(MID).eval(); proc=CLIPProcessor.from_pretrained(MID)
    import csv, random
    rows=[[x.strip() for x in r] for r in list(csv.reader(open("dataset/AMT-testset/data.csv")))[1:]]
    amt=sorted(set(f for r in rows for f in r[1:4])); rest=sorted(set(os.listdir("dataset/taglabel"))-set(amt)); random.Random(0).shuffle(rest)
    keys=amt+rest
    out={}; t=time.time(); batch=[]
    def flush():
        with torch.no_grad():
            px=proc(images=[b[1] for b in batch],return_tensors="pt")["pixel_values"]
            e=model.visual_projection(model.vision_model(pixel_values=px).pooler_output)
        for (k,_),v in zip(batch,e.numpy()): out[k]=v
        batch.clear()
    with Pool(1) as pool:
        for n,(k,im) in enumerate(pool.imap(job,keys,chunksize=32)):
            if im is not None: batch.append((k,im))
            if len(batch)==64: flush()
            if n%1000==0: print(n,len(out),f"{time.time()-t:.0f}s",flush=True); pickle.dump(out,open("clip_myfonts.pkl","wb"))
    if batch: flush()
    pickle.dump(out,open("clip_myfonts.pkl","wb")); print("done",len(out))
