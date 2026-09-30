# Checks configs/tagVocabulary.json before it's used for training or search:
#   1. merges: ROC-AUC of each canonical tag (label = OR of its member tags) vs. its best single member,
#      same frozen-ViT + MLP probe and official MyFonts splits as e22_vocab.py. A merge that costs a lot of
#      AUC is joining tags that don't look alike.
#   2. alias collisions: one phrase mapping to several canonicals at weight >= 0.8.
#   3. parses of the 30 trial queries (experiments/choice-searches/estimateSearchQuality.py) + unmatched words.
# Run from the working dir holding labels.pkl / cache/ / dataset/ (as e22_vocab.py):
#   python <repo>/experiments/critical-review/validate_tag_vocabulary.py --repo <repo>
import argparse, json, os, pickle, re, sys, importlib.util
import numpy as np, torch
from collections import Counter
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--noProbe", action="store_true")
p.add_argument("--cache", default="cache")
p.add_argument("--dataset", default="dataset")
args = p.parse_args()

spec = importlib.util.spec_from_file_location("tagVocabulary", os.path.join(args.repo, "utils", "tagVocabulary.py"))
tv = importlib.util.module_from_spec(spec); spec.loader.exec_module(tv)
vocab = tv.TagVocabulary(os.path.join(args.repo, "configs", "tagVocabulary.json"))
C = vocab.canonical

print("== alias collisions (weight >= 0.8 to more than one canonical) ==")
n = 0
for phrase, targets in sorted(vocab.aliases.items()):
    strong = [(c, w) for c, w in targets if w >= 0.8]
    if len(strong) > 1:
        print(f"  {' '.join(phrase)!r}: {strong}"); n += 1
print(f"  {n} collisions")

print("\n== trial queries ==")
src = open(os.path.join(args.repo, "experiments", "choice-searches", "estimateSearchQuality.py")).read()
queries = eval(re.search(r"QUERIES = (\[.*?\])", src, re.S).group(1))
allUnmatched = Counter(); empty = 0
for q in queries:
    w, un = vocab.parse(q)
    allUnmatched.update(un); empty += not any(v > 0 for v in w.values())
    ws = ", ".join(f"{c}:{v:+.1f}" for c, v in sorted(w.items(), key=lambda x: -abs(x[1])))
    print(f"  {q!r}\n      -> {ws or '(none)'}" + (f"   unmatched: {un}" if un else ""))
print(f"  queries with no positive tag: {empty}/{len(queries)}; unmatched words: {dict(allUnmatched.most_common())}")

if args.noProbe: sys.exit()

print("\n== merged-label AUC vs best member (frozen ViT + MLP probe, test+val) ==")
torch.manual_seed(0)
lab, _ = pickle.load(open("labels.pkl", "rb"))
keys = pickle.load(open(os.path.join(args.cache, "all_keys.pkl"), "rb")); X = np.load(os.path.join(args.cache, "all_X.npy")); E = dict(zip(keys, X))
rd = lambda f: [l.strip() for l in open(f) if l.strip()]
tr = [k for k in rd(os.path.join(args.dataset, "fontset", "trainset")) if k in E]
te = [k for k in rd(os.path.join(args.dataset, "fontset", "testset")) + rd(os.path.join(args.dataset, "fontset", "valset")) if k in E]
canons = list(C); ci = {c: i for i, c in enumerate(canons)}
memberOf = {m: c for c in canons for m in C[c]["members"]}
def Y(ks):
    y = np.zeros((len(ks), len(canons)), np.float32)
    for i, k in enumerate(ks):
        for t in lab[k][1]:
            if t in memberOf: y[i, ci[memberOf[t]]] = 1
    return y
A = np.stack([E[k] for k in tr]); mu = A.mean(0); sd = A.std(0) + 1e-6
f = lambda M: torch.tensor(((M - mu) / sd).astype(np.float32))
net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(512, 1024), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                          torch.nn.Linear(1024, len(canons)))
opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); Xt = f(A); Yt = torch.tensor(Y(tr))
for ep in range(40):
    net.train(); perm = torch.randperm(len(Xt))
    for s in range(0, len(Xt), 256):
        b = perm[s:s + 256]
        l = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]), Yt[b]); opt.zero_grad(); l.backward(); opt.step()
net.eval()
with torch.no_grad(): P = net(f(np.stack([E[k] for k in te]))).numpy()
Yte = Y(te)
rows = []
for c in canons:
    j = ci[c]
    if Yte[:, j].sum() < 5: continue
    a = roc_auc_score(Yte[:, j], P[:, j]); best = C[c]["bestMemberAuc"]
    rows.append((a - best, c, a, best, len(C[c]["members"]), int(Yte[:, j].sum())))
rows.sort()
aucs = np.array([r[2] for r in rows])
print(f"  {len(rows)} canonicals scored; median AUC {np.median(aucs):.3f}; frac >= 0.75: {np.mean(aucs >= 0.75):.2f}")
print("  merges losing >= 0.03 AUC vs best member (delta, canonical, merged, best, #members, #pos):")
for r in rows:
    if r[4] > 1 and r[0] <= -0.03: print(f"    {r[0]:+.3f}  {r[1]:<18} {r[2]:.3f} {r[3]:.3f} {r[4]:>2} {r[5]:>4}")
print("  lowest-AUC canonicals:")
for r in sorted(rows, key=lambda r: r[2])[:12]: print(f"    {r[1]:<18} {r[2]:.3f}  (#pos {r[5]})")
json.dump({r[1]: round(r[2], 4) for r in rows}, open("tagVocabulary_auc.json", "w"), indent=1)
