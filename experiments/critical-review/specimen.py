import numpy as np
from PIL import Image, ImageFont
CH=list("ABCGQR")+list("abcdefghijklmnopqrstuvwx")
def _ink_myfonts(k,c):
    fn=c*2 if c.isupper() else c
    a=np.array(Image.open(f"dataset/fontimage/{k}_{fn}.png").convert("L"),dtype=np.float32)
    g=1-a/255.0; cols=np.where(g.max(0)>0.05)[0]; rows=np.where(g.max(1)>0.05)[0]
    if len(cols)==0: return None
    return g[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
def _ink_file(p,c,px=200):
    f=ImageFont.truetype(p,px); m=f.getmask(c)
    if m.size==(0,0): return None
    g=np.asarray(Image.Image()._new(m),dtype=np.float32)/255.0
    cols=np.where(g.max(0)>0.05)[0]; rows=np.where(g.max(1)>0.05)[0]
    if len(cols)==0: return None
    return g[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
def specimen(k=None,path=None,size=224,cols=6,rows=5):
    gl=[_ink_myfonts(k,c) if path is None else _ink_file(path,c) for c in CH]
    if sum(g is None for g in gl)>6: return None
    cell=size//cols; hmax=max(max(g.shape) for g in gl if g is not None)
    s=cell*0.85/hmax
    canvas=Image.new("L",(size,size),255)
    for i,g in enumerate(gl):
        if g is None: continue
        im=Image.fromarray((255*(1-g)).astype(np.uint8)); w,h=max(1,round(g.shape[1]*s)),max(1,round(g.shape[0]*s))
        im=im.resize((w,h),Image.LANCZOS); r,c=divmod(i,cols)
        canvas.paste(im,(c*cell+(cell-w)//2, r*(size//rows)+((size//rows)-h)//2))
    return canvas.convert("RGB")
