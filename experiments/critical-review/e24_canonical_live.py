# Second blind trial: the canonical-vocabulary tag search (C) against the previous best (B) and random (R),
# over the same DaFont gallery and the same 30 queries as e20_live.py.
#   B = e20's system: raw MyFonts tags (>=20 train fonts), regex query parser, z-scored tagger over the
#       MyFonts-style re-render (dafont_mf.pkl). Re-run here with the same seed/recipe.
#   C = canonical vocabulary (configs/tagVocabulary.json): tagger trained on merged labels (OR of members),
#       utils/tagVocabulary.py parser (aliases, weights, scoped negation), z-scored scores, same re-render,
#       DaFont "Dingbats" category removed from the gallery.
#   R = 8 random gallery fonts.
# Results are pooled per query (union, deduplicated, shuffled) so each font is judged once, TRECVID-style.
# Run from the work dir holding labels.pkl, all_sources.pkl, fontname_map.pkl, dafont_mf.pkl, tiles/:
#   python <repo>/experiments/critical-review/e24_canonical_live.py --repo <repo> --cache <dir with all_X.npy>
import argparse, csv, importlib.util, json, os, pickle, re, zlib
import numpy as np, torch
from collections import Counter
from PIL import Image, ImageFont, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--cache", default="cache")
p.add_argument("--dataset", default="dataset")
p.add_argument("--dafont", default="dafont")
p.add_argument("--k", type=int, default=8)
args = p.parse_args()

spec = importlib.util.spec_from_file_location("tagVocabulary", os.path.join(args.repo, "utils", "tagVocabulary.py"))
tv = importlib.util.module_from_spec(spec); spec.loader.exec_module(tv)
vocabC = tv.TagVocabulary(os.path.join(args.repo, "configs", "tagVocabulary.json"))

src_q = open(os.path.join(args.repo, "experiments", "choice-searches", "estimateSearchQuality.py")).read()
QUERIES = eval(re.search(r"QUERIES = (\[.*?\])", src_q, re.S).group(1))
lab, _ = pickle.load(open("labels.pkl", "rb"))
keys = pickle.load(open(os.path.join(args.cache, "all_keys.pkl"), "rb")); X = np.load(os.path.join(args.cache, "all_X.npy"))
E = dict(zip(keys, X))
src = pickle.load(open("all_sources.pkl", "rb")); fm = pickle.load(open("fontname_map.pkl", "rb"))
MF = pickle.load(open("dafont_mf.pkl", "rb"))
rd = lambda f: [l.strip() for l in open(f) if l.strip()]
tr = [k for k in rd(os.path.join(args.dataset, "fontset", "trainset")) + rd(os.path.join(args.dataset, "fontset", "valset")) if k in E]
Xtr = np.stack([E[k] for k in tr]); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
f = lambda A: torch.tensor(((A - mu) / sd).astype(np.float32))

def trainTagger(Y):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(512, 1024), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                              torch.nn.Linear(1024, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); Xt = f(Xtr); Yt = torch.tensor(Y)
    for ep in range(40):
        net.train(); perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 256):
            b = perm[s:s + 256]
            l = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]), Yt[b]); opt.zero_grad(); l.backward(); opt.step()
    return net.eval()

# --- B: raw-tag system, as in e20_live.py
cnt = Counter(t for k in tr for t in lab[k][1]); vocabB = [t for t, c in cnt.items() if c >= 20]; vi = {t: i for i, t in enumerate(vocabB)}
YB = np.zeros((len(tr), len(vocabB)), np.float32)
for i, k in enumerate(tr):
    for t in lab[k][1]:
        if t in vi: YB[i, vi[t]] = 1
netB = trainTagger(YB)
idf = np.log(len(tr) / np.maximum(YB.sum(0), 1)); STOP = {"font", "fonts", "typeface", "typefaces"}
tagRe = [re.compile(r"\b" + re.escape(t.replace("-", " ")) + r"s?\b") for t in vocabB]
def parseB(q):
    ql = q.lower().replace("-", " "); w = np.zeros(len(vocabB), np.float32)
    for j, r in enumerate(tagRe):
        if vocabB[j] not in STOP and r.search(ql): w[j] = np.log1p(idf[j])
    return w

# --- C: canonical system
canons = list(vocabC.canonical); ci = {c: i for i, c in enumerate(canons)}
memberOf = {m: c for c in canons for m in vocabC.canonical[c]["members"]}
YC = np.zeros((len(tr), len(canons)), np.float32)
for i, k in enumerate(tr):
    for t in lab[k][1]:
        if t in memberOf: YC[i, ci[memberOf[t]]] = 1
netC = trainTagger(YC)

# DaFont category per gallery font (for the dingbat filter)
catByFile = {}
with open(os.path.join(args.dafont, "info.csv")) as fh:
    for row in csv.DictReader(fh): catByFile[row["filename"]] = row["category"]
def category(k):
    for s, path in fm.get(k, []):
        if s == "dafont" and os.path.basename(path) in catByFile: return catByFile[os.path.basename(path)]
    return None

gallery = [k for k in keys if src[k] == {"dafont"} and k in MF]
def Z(net, vecs):
    with torch.no_grad(): S = net(f(vecs)).numpy()
    return (S - S.mean(0)) / (S.std(0) + 1e-6)
GB = np.stack([MF[k] for k in gallery])
ZB = Z(netB, GB); ZC = Z(netC, GB)
notDingbat = np.array([category(k) != "Dingbats" for k in gallery])
print(f"gallery {len(gallery)}; dingbat-category fonts removed for C: {(~notDingbat).sum()}")

os.makedirs("tiles", exist_ok=True)
def tile(k):
    fn = f"tiles/{zlib.crc32(k.encode())}.png"
    if os.path.exists(fn): return fn
    im = Image.new("L", (360, 56), 255); d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(fm[k][0][1], 30); d.text((8, 8), "Handgloves Quiz", font=font, fill=0)
    except Exception: d.text((8, 20), "(render failed)", fill=0)
    im.save(fn, optimize=True); return fn

out = []
for q in QUERIES:
    wB = parseB(q)
    weights, unmatched = vocabC.parse(q)
    wC = np.zeros(len(canons), np.float32)
    for c, w in weights.items(): wC[ci[c]] = w
    rec = {"query": q, "tagsB": [vocabB[i] for i in np.nonzero(wB)[0]],
           "tagsC": [f"{c} {w:+.1f}" for c, w in sorted(weights.items(), key=lambda x: -abs(x[1]))], "unmatchedC": unmatched}
    rec["B"] = [gallery[i] for i in np.argsort(-(ZB @ wB))[:args.k]] if wB.any() else []
    if (wC > 0).any():
        s = ZC @ wC; s[~notDingbat] = -np.inf
        rec["C"] = [gallery[i] for i in np.argsort(-s)[:args.k]]
    else: rec["C"] = []
    rec["R"] = [gallery[i] for i in np.random.RandomState(zlib.crc32(q.encode())).choice(len(gallery), args.k, False)]
    rec["tiles"] = {k: tile(k) for s_ in ("B", "C", "R") for k in rec[s_]}
    out.append(rec)
    print(f"{q!r}: C={rec['tagsC']}  overlap(B,C)={len(set(rec['B']) & set(rec['C']))}")
json.dump({"gallery": len(gallery), "results": out}, open("e24_results.json", "w"), indent=1)
