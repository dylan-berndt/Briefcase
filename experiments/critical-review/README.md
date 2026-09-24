# Critical review: why text→font retrieval has been stuck

A search for structural flaws and wrong assumptions, not for small tuning wins. Every number
below was measured in this session on CPU against the real data: the ICCV-2019 MyFonts dataset, a
fresh clone of `google/fonts`, `dafonts-free`, `embeddings/all.json`, and
`checkpoints/pretrain/best`. Where a number depends on an estimator choice, the choice is stated.

## Setup / reproducing

Run the scripts from the repo root. They expect `dataset/`, `google/fonts/`, and `dafont/` laid
out as `initialize.sh` produces, and write their intermediate pickles to the current directory.
One prep step builds the `cache/` inputs:

```python
import json, numpy as np, pickle, os; os.makedirs("cache", exist_ok=True)
d = json.load(open("embeddings/all.json")); k = list(d)
np.save("cache/all_X.npy", np.array([d[x] for x in k], np.float32)); pickle.dump(k, open("cache/all_keys.pkl", "wb"))
```

Then run `names.py` (maps `all.json` keys to their source font files: 18,711 MyFonts, 17,018
DaFont, 3,595 Google, 77 in both DaFont and Google, 20 unresolved) and `labels.py` (per-font
source labels). `render.py` holds exact copies of the two rendering pipelines. `model.py` is a
standalone copy of the ViT.

`verify.py` and `verify2.py` confirm the renders reproduce `all.json` exactly (cos = 1.0000 once
the case bug below is accounted for). Everything else is built on those renders.

## Findings

### 1. What the benchmark actually measures: exact-font recall on tag-derived captions

The captions (`generate.py`, `generateV2.py`) are written by an LLM that never sees the font. It
only gets the source labels:
- MyFonts: foundry tags.
- Google: tags plus the "about" text.
- DaFont: only `category` and `theme`, i.e. **2 labels per font**.

The text therefore carries the tags' information and nothing else. How much of that information
is visual decides what any image model can reach.

**Text→text retrieval** (`e1_textretrieval.py`: one held-out caption vs. the font's other
captions, TF-IDF, no vision at all):

| captions | R@1 | R@10 | median rank |
|---|---|---|---|
| V1, all 25k fonts | 23% | 43% | 21 |
| V1, MyFonts | 27% | 50% | 10 |
| V1, DaFont | 1% | 4% | 636 |

For DaFont, 463 fonts on average share an identical `(category, theme)` pair (`e1_ceiling.py`).
The text-identity ceiling for DaFont is ~2% R@1 by construction.

**The decomposition** (`e17_fullscale.py`, `e17b.py`, `e17c.py`): all 18,710 MyFonts fonts with
captions, leave-one-out, one held-out caption per font. Tags are split by how visually decodable
they are, using a linear probe AUC from `e10_tagauc.py`.

| font representation used to predict its captions | estimator | R@1 | R@10 | median rank |
|---|---|---|---|---|
| its own other captions (oracle) | TF-IDF | 27.0% | 50.8% | 10 |
| its true tags, all 400 frequent | ridge→LSA | 22.7% | 49.8% | 11 |
| its true tags, AUC≥.75 (283 tags) | ridge→LSA | 15.0% | 37.8% | 25 |
| its true tags, AUC≥.85 (105 strongly visual tags) | ridge→LSA | 3.9% | 12.7% | 220 |
| same, kNN k=5..100 | kNN | 1.1–1.7% | 7.9–9.4% | 227–305 |
| **ViT `all.json`** | ridge / MLP | 0.2–0.3% | 2.6–3.0% | ~700 |
| ViT `all.json` | kNN | 0.1% | 1.8% | 884 |

What this shows:
- Most of the identity information in the captions sits in tags that are weakly visual or not
  visual at all ("travel", "film", "label", "ad", decades…). No image model can recover it.
- Even perfect knowledge of the strongly visual tags gives only R@10 ≈ 9–13% at 18.7k fonts.
  That is the regime every approach in this repo has landed in.
- The ViT itself captures only part of even that ceiling. The ceiling depends on where the
  visual/non-visual line is drawn (table), so treat it as a band, not a hard number.

### 2. The backbone is not the main bottleneck; it is near published state of the art on relevance metrics

**ICCV-2019 MyFonts-test protocol** (`e13_iccv.py`: frozen ViT features plus a small MLP tag
head, official split, mAP / NDCG over the 1,866 test fonts):

| | single tag (300) | single tag (full) | multi tag |
|---|---|---|---|
| RelationNet (ICCV table) | 15.3 / 57.5 | 5.7 / 29.3 | 7.5 / 28.1 |
| **project ViT + MLP head** | **20.5 / 62.8** | **11.4 / 36.9** | **11.0 / 32.1** |
| ICCV full model (end-to-end ResNet-50, 128px) | 27.8 / 69.8 | 17.8 / 43.7 | 15.8 / 36.4 |

**AMT-test** (`e7_amt.py`): 1,661 human "which of 3 fonts best fits tag X" triplets, where all 3
fonts already carry the tag.

| method | accuracy |
|---|---|
| chance | 33.3% |
| always picking the most common answer position | 38.7% |
| CLIP B/16 zero-shot | 40.6% |
| ICCV basic model | 44.5% |
| **project ViT probe** | **46.5%** |
| ICCV full model | 47.5% |
| CLIP B/16 probe | 50.2% |

**CLIP B/16** on rendered specimen sheets, same fonts, same protocol (`clip_embed.py`,
`e15_knn_cmp.py`), is only slightly better than the ViT:
- Caption kNN on a 2,000-font gallery: R@10 14.8% vs. 12.6% for the ViT; the text oracle gets
  75.8%.
- ICCV single-tag (300) mAP: 22.0 vs. 19.7.

A far larger, web-trained vision model does not change the regime.

**Metric note** (`e16_batchmap.py`): the old `retrieval.py` (commit 8b28fb1) logged mAP computed
*inside 256-item batches*. For the same predictions that reads about 2× the corpus-level mAP:
21.6 in-batch vs. 11.4 over the full test set, with random at 3.6 vs. 1.3.

### 3. The visual space separates fonts extremely well; "corpus density" is not the problem

**Identity test** (`e9_identity.py`: 4,962 MyFonts fonts, per-glyph embeddings from
`checkpoints/pretrain/best`). The query is a font's mean embedding over 13 lowercase letters. The
gallery is every font's mean over the *other* 13 letters.

| query glyphs | R@1 | R@10 |
|---|---|---|
| 13 letters | **94.1%** | 99.8% |
| 3 letters | 61.7% | 92.0% |
| 1 letter | 36.4% | 72.3% |

Caveat: this checkpoint trained on nearly all of these fonts, so the test is in-distribution.
It still shows that 0.96 nearest-neighbor cosine and near-perfect separability coexist. NN
cosine is not a discriminability measure.

**The "~3–4 participating dimensions in pixels" figure** (`e11_pr.py`) reproduces only when the
mean is *not* subtracted:

| data | uncentered PR | centered PR | dims for 90% variance | 1-NN accuracy |
|---|---|---|---|---|
| font glyphs | 3.3 | 15–18 | 78–108 | — |
| MNIST (reference) | 5.0 | 30 | 83 | 92% |

Properly centered, font glyphs are in the same regime as MNIST, which is easily separable.

**Italic and weight are linearly decodable** despite the ±25° rotation augmentation
(`e6_attrs.py`): family-grouped AUC 0.976 for italic and 0.971 for bold vs. light.

### 4. Rendering pipeline confound (real, fixable)

MyFonts glyphs go through `loadRochesterImage`, which scales **each glyph** to fill a 32px box.
Google and DaFont glyphs are rendered at font-metric size (`imagesFromFont`).

The source is linearly predictable from `all.json` (`e3_source.py`):
- MyFonts vs. other: 97.8% (chance 66.7%).
- 3-way, balanced: 89% (chance 33%).
- 82% of each font's 10 nearest neighbors share its source (chance 33%).

**Controlled test** (`e14_pipeline.py`, `e14b.py`): 1,482 DaFont fonts, each rendered through
both pipelines.

| query vs. gallery | R@1 |
|---|---|
| same pipeline, 13 letters vs. the *other* 13 | **98.7%** |
| same 26 letters, pipeline swapped | **37.6%** |

Changing the renderer moves a font further than changing every letter.

Tag probes trained on MyFonts-pipeline images (the only human-tagged data) transfer worse to
metric-rendered fonts. AUC on DaFont's own labels, metric render → MyFonts-style render:

| DaFont label → probe tag | metric | MyFonts-style |
|---|---|---|
| handwrite | 0.859 | 0.963 |
| script | 0.844 | 0.927 |
| blackletter | 0.842 | 0.912 |
| comic | 0.667 | 0.929 |
| calligraphy | 0.859 | 0.985 |

10 of 11 labels are better with matched rendering. 53% of the corpus (DaFont + Google) sits in
the other domain from all the human tag supervision.

### 5. Bugs and data issues (verified; smaller impact)

- **Case-mixing bug.** `collectFontSetPaths` / `loadMyFontsImagePaths` key glyphs by
  `stem[-2]`, which is the same letter for `al` and `au`. `generateEmbeddings` therefore builds
  each `all.json` vector from an arbitrary per-letter mix of UPPER and lower case, whichever file
  the glob returned last. Reproduced exactly (`verify2.py`: cos 1.0000 with the fitted mix). Some
  fonts are 100% uppercase, some 100% lowercase, some mixed.
  - Identity impact: an `all.json` vector matches its own true lowercase vector only 59% R@1
    among 4,962.
  - Tag AUC impact is small: 0.786 → 0.791 (`e18_casefix.py`).
  - Fix: key by the full `{letter}{case}` suffix.
- **Prefix mis-pairing.** The longest-prefix trie pairs ~940 (V1) to ~1,560 (V2) corpus entries
  with the wrong family's captions, e.g. `'3'` → `'32 pages Regular'`, `'Acid'` →
  `'AcidDreamer Regular'` (`e5_siblings.py`).
- **Undescribed distractors.** 11.6k (V1) to 17.5k (V2) of the 39,421 corpus entries have no
  description at all. They are pure distractors in every 39k-scale eval.
- **Style siblings.** Google and DaFont families have several style entries that share one
  description, but only the shortest name counts as the target. That caps R@1 at ~0.80–0.87 for
  those sources.
- **Non-visual Google captions.** Generated Google captions often describe non-visual metadata:
  script/language support, glyph counts, OpenType features (`e8_halluc.py`). Coarse-category
  contradictions are rare, e.g. "serif" appears in 3% of captions for sans fonts, so gross
  hallucination is **not** a major issue.
- **Label-contradicting finetune augmentation (read from code, not measured).** The finetune
  loader (`CombinedQueryData`) uses `RandomResizedCrop(ratio=(0.75, 1.333))` and
  `RandomRotation(25)`. The captions say "condensed", "extended", "italic", so these
  augmentations randomize the very attributes the text asserts. The pretraining loader uses
  ratio (1, 1).

### 6. Tag-bottleneck search engine: what actually helps (`e19_search.py`)

Setup: query → tag weights; font vector → tag logits from an MLP tagger trained on MyFonts train
fonts only; rank by the weighted sum of per-tag z-scores. Three query parsers:
- **lexical**: query words phrase-matched to tag names, `log1p(idf)` weights (approximates
  TagSearch, without spaCy synonym vectors);
- **learned kNN**: tag distribution of the 50 most similar MyFonts *training* captions, as lift
  over the tag prior;
- **learned linear**: TF-IDF → tags.

| eval | lexical | learned kNN | learned linear | lexical + kNN |
|---|---|---|---|---|
| A. ICCV multi-tag queries as text (1000), MyFonts test, mAP | **10.2** | 5.8 | 1.3 | 7.1 |
| B. held-out natural captions (1582), top-10 visual-tag Jaccard (random 0.041) | **0.142** | 0.123 | 0.056 | 0.127 |
| B. same, exact-font R@10 in 1866 | **11.4%** | 7.8% | 1.1% | 8.6% |
| C. DaFont, its 36 category/theme names as queries, metric render, mAP | 13.3 | 10.3 | 5.4 | 11.3 |
| C. DaFont, same fonts **MyFonts-style render**, mAP | **26.5** | 21.0 | 6.1 | 24.3 |

Section A: for single-tag queries, lexical 12.5 vs. oracle tags 12.8. For multi-tag, lexical
equals the oracle (10.2 vs. 10.0).

Read:
- **Learning query→tag from the generated captions did not beat lexical matching**, even on
  natural-language queries. The hypothesis is rejected. (The linear model is likely
  undertrained, but the kNN version is a fair test.)
- **The biggest measured lever for searching DaFont with MyFonts-trained models is rendering.**
  Same fonts, same models, same queries: mAP 13.3 → 26.5, P@10 0.20 → 0.34.
- Caveat: DaFont's labels are coarse and noisy, so section C is a transfer sanity check, not a
  precise score. Primary validation is the MyFonts test split (A, B).

## What this implies (recommendations, untested unless stated)

1. **Stop using exact-font R@k on LLM captions as the success metric.** With these labels even a
   perfect visual tagger stays near R@10 ≈ 10% at 18.7k fonts. Use relevance metrics instead:
   ICCV mAP/NDCG and AMT. On those, the existing backbone is already between the 2019 baselines
   and SOTA.
2. **Unify rendering, in the only direction possible.** The MyFonts dataset ships only glyph
   PNGs, no font files. So "matched rendering" here means rendering DaFont/Google *font files*
   through a simulation of the MyFonts pipeline (`render.myfontsStyleFromFont`: large render,
   tight crop, then the exact `loadRochesterImage` scale-to-fit), not the reverse.
   - A myfonts-vs-dafont source classifier trained on `all.json` labels 1.6% of metric-rendered
     DaFont fonts as MyFonts, versus 86.5% of the same fonts rendered this way.
   - Measured effect on tag transfer: up to +0.26 AUC. Measured effect on search: see finding 6.
3. **Text search: use text where the text is.** For fonts with human metadata (MyFonts tags,
   Google tags/descriptions), plain TF-IDF text→text beats every visual route by an order of
   magnitude on these queries. This is optimistic, since the queries are derived from that same
   metadata. For metadata-poor fonts (DaFont), fall back to predicted *visual* tags.
4. **Known-item search needs a visual channel, not a better text model.** The visual embedding
   identifies a font from 1–3 glyphs at 36–62% R@1 among ~5k fonts. Query-by-image, or
   "more like this" once the user has *any* close font, plays to that strength.
5. **If richer text is wanted, caption from pixels.** Have a VLM describe rendered specimens
   instead of paraphrasing tags. This is the only way to put more *visual* bits into the text
   side.
