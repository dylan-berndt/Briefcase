import torch, torch.nn as nn, json, numpy as np
torch.set_num_threads(4)
class PatchEmbedding(nn.Module):
    def __init__(s,c):
        super().__init__(); s.projection=nn.Conv2d(1,c["embedDim"],c["patchSize"],c["patchSize"])
        s.positional=nn.Parameter(torch.zeros(1,(c["imageSize"]//c["patchSize"])**2,c["embedDim"]))
    def forward(s,x):
        x=s.projection(x).flatten(2).transpose(1,2); return x+s.positional[:,:x.size(1)]
class ViT(nn.Module):
    def __init__(s,c,nclass):
        super().__init__(); s.patching=PatchEmbedding(c); s.clsToken=nn.Parameter(torch.randn(1,1,c["embedDim"]))
        s.transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(c["embedDim"],c["heads"],c["embedDim"]*3,batch_first=True),num_layers=c["layers"])
        s.transpose=nn.ConvTranspose2d(c["embedDim"],1,c["patchSize"],c["patchSize"])
        s.classifier=nn.Sequential(nn.Linear(c["embedDim"],c["embedDim"]),nn.ReLU(),nn.Linear(c["embedDim"],nclass))
    def forward(s,x):  # x [B,H,W]
        x=s.patching(x.unsqueeze(1)); x=torch.cat([s.clsToken.expand(x.shape[0],-1,-1),x],1)
        return s.transformer(x)[:,0]
def load(path="checkpoints/pretrain/best"):
    c=json.load(open(path+"/config.json"))["model"]
    sd=torch.load(path+"/checkpoint.pt",map_location="cpu",weights_only=False)
    if hasattr(sd,"state_dict"): sd=sd.state_dict()
    m=ViT(c,sd["classifier.2.weight"].shape[0]); m.load_state_dict(sd); m.eval(); return m
@torch.no_grad()
def embedGlyphs(m,imgs,bs=512):
    out=[]
    for i in range(0,len(imgs),bs):
        out.append(m(torch.tensor(np.stack(imgs[i:i+bs]),dtype=torch.float32)))
    return torch.cat(out).numpy()
def fontVec(E):  # all.json convention: normalize per glyph, mean
    E=E/np.linalg.norm(E,axis=1,keepdims=True); return E.mean(0)
