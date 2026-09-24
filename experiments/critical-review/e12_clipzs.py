import torch, numpy as np, pickle, csv
from transformers import CLIPModel, CLIPTokenizer
torch.set_num_threads(1)
MID="openai/clip-vit-base-patch16"; m=CLIPModel.from_pretrained(MID).eval(); tok=CLIPTokenizer.from_pretrained(MID)
E=pickle.load(open("clip_myfonts.pkl","rb"))
rows=[[x.strip() for x in r] for r in list(csv.reader(open("dataset/AMT-testset/data.csv")))[1:]]
tags=sorted(set(r[0] for r in rows)); T={}
with torch.no_grad():
    for tpl in ["a {} font","a {} typeface","{}"]:
        t=tok([tpl.format(x.replace("-"," ")) for x in tags],padding=True,return_tensors="pt")
        v=m.text_projection(m.text_model(**t).pooler_output).numpy(); v/=np.linalg.norm(v,axis=1,keepdims=True)
        for x,vv in zip(tags,v): T.setdefault(x,[]).append(vv)
c=0;n=0
for r in rows:
    if not all(f in E for f in r[1:4]): continue
    tv=np.mean(T[r[0]],0); s=[E[f]@tv/np.linalg.norm(E[f]) for f in r[1:4]]
    c+=int(np.argmax(s)==int(r[4])); n+=1
print(f"CLIP B/16 zero-shot AMT acc={c/n:.3f} (n={n})")
