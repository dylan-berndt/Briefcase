import os, numpy as np, cv2
from PIL import Image, ImageFont
from fontTools.ttLib import TTFont

def metricRender(fontPath, char, fontSize=32):
    """Exact copy of utils/loaders/standard.imagesFromFont for one char (google/dafont pipeline)."""
    imageSize=int(fontSize*1.5)
    font = ImageFont.truetype(fontPath, fontSize)
    ascent, descent = font.getmetrics()
    standard = fontSize * (fontSize / ascent)
    font = ImageFont.truetype(fontPath, standard)
    mask = font.getmask(char); box = font.getbbox(char)
    if mask.size == (0, 0): return None
    im = Image.Image()._new(mask)
    canvas = Image.new("L", (imageSize, imageSize), 0)
    baseline = int(imageSize * 0.9); offset = baseline - ascent
    canvas.paste(im, ((imageSize - im.width) // 2, offset + box[1] - int(imageSize * 0.25)))
    # saved as .bmp then read via cv2 grayscale /255
    return np.asarray(canvas,dtype=np.float32)/255.0

def rochesterFromGray(gray, fontSize=32):
    """Exact copy of utils/loaders/myfonts.loadRochesterImage after inversion (gray: glyph=1)."""
    imageSize=int(fontSize*1.5); padding=8; target=imageSize-2*padding
    col_max=gray.max(0); nz=np.where(col_max>0.05)[0]
    if len(nz)==0: return None
    gray=gray[:, :nz[-1]+1]
    row_max=gray.max(1); nz=np.where(row_max>0.05)[0]
    if len(nz)==0: return None
    gray=gray[nz[0]:nz[-1]+1,:]
    h,w=gray.shape; s=min(target/h,target/w)
    nh,nw=max(1,round(h*s)),max(1,round(w*s))
    r=cv2.resize(gray,(nw,nh),interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR)
    c=np.zeros((imageSize,imageSize),np.float32); y0=(imageSize-nh)//2; x0=(imageSize-nw)//2
    c[y0:y0+nh,x0:x0+nw]=r
    # saved as bmp uint8 and reloaded
    return (np.clip(c*255,0,255).astype(np.uint8)).astype(np.float32)/255.0

def rochesterFile(path):
    a=np.array(Image.open(path).convert("RGB"),dtype=np.float32)
    return rochesterFromGray(1.0-a[:,:,0]/255.0)

def myfontsStyleFromFont(fontPath, char, px=200):
    """Simulate a MyFonts-dataset source image from a font file: large render, glyph = ink."""
    font=ImageFont.truetype(fontPath,px); mask=font.getmask(char)
    if mask.size==(0,0): return None
    g=np.asarray(Image.Image()._new(mask),dtype=np.float32)/255.0
    return rochesterFromGray(g)
