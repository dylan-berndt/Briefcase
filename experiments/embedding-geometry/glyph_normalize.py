"""
Torch-free home of centerAndScaleNormalize, so pipelines without torch (experiments/internet-archive/) can use the
exact same per-image normalization as ssim_duplicates.py and the raw-pixel scripts that import it from there.
"""
import numpy as np
from scipy.ndimage import affine_transform


def centerAndScaleNormalize(img, outSize=48):
    """Per-IMAGE normalization (no pairing involved): recenter on the image's own ink centroid and
    rescale so its own ink bounding box fills a fixed fraction of the canvas. This is what makes the
    cheap prefilter shift/scale-INVARIANT -- two accidentally-shifted/scaled copies of the same glyph
    map to nearly the same normalized image, without ever comparing them to each other."""
    mass = img.sum()
    if mass < 1e-6:
        return img
    ys, xs = np.nonzero(img > 0.05)
    if len(xs) == 0:
        return img
    cy, cx = ys.mean(), xs.mean()
    h = max(ys.max() - ys.min(), 1)
    w = max(xs.max() - xs.min(), 1)
    scale = (outSize * 0.6) / max(h, w)   # target ink extent ~60% of canvas, same for every glyph
    matrix = np.eye(2) / scale
    center = np.array([outSize / 2, outSize / 2])
    offset = np.array([cy, cx]) - matrix @ center
    return affine_transform(img, matrix, offset=offset, output_shape=(outSize, outSize), order=1, mode="constant", cval=0.0)
