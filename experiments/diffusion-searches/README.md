# Text -> Visual Font Embedding Diffusion Experiment

Tests whether a diffusion model can learn `P(visual font embedding | text embedding)`.

## Current pipeline (targets `embeddings/all.json`, the staging-branch corpus)

```
experiments/embed_text_local.py  # -> embeddings/sentenceQueries_<model>.pkl  {fontKey: [nQueries, dim]}
                                  #    run LOCALLY (not this sandbox) -- see Compute note
experiments/train.py             # -> checkpoints/diffusion/{checkpoint.pt, whitener.npz, config.json, test_pairs.json}
experiments/evaluate.py          # recall@k over held-out queries
experiments/diagnose_conditioning.py     # generated-vs-true pairwise similarity structure
experiments/diagnose_text_ablation.py    # denoising loss with real vs. shuffled text, by noise level
experiments/diagnose_manifold_realism.py # does generation look like ANY real font, right or wrong
```

`embeddings/all.json` (39,421 fonts, LFS-tracked) and `results/fontQueriesV2.json`
(18,940 fonts x up to 16 Gemma-generated queries each) already exist on `staging`
and are consumed directly by `experiments/dataset.py` -- no font rasterization or
ViT inference needed for this path.

## Design decisions / assumptions

- **Visual embedding**: `embeddings/all.json`, not the raw pretrained ViT CLS
  token an earlier version of this experiment targeted (`embeddings/font_vit_embeddings.json`,
  still around for the effective-dimension comparison below). Note: `all.json`
  is *not* the `ViTEmbedder` contrastive-finetuning output -- that's a separate
  file, `embeddings/allText.json` -- `all.json` is on the raw/pretrain side too
  (per the user, from the newer 12-layer pretrain checkpoint), just a better
  model than the one `font_vit_embeddings.json` used. `embed_fonts.py` /
  `effective_dimension.py` / `character_variance.py` / `pooling_bug_check.py`
  still exist and operate on `font_vit_embeddings.json` for that comparison,
  but are no longer part of the main training path.
- **Text encoder**: swappable via `--model`/`--sentenceModel` (must match between
  `embed_text_local.py` and `train.py`/`evaluate.py`), defaulting to
  `BAAI/bge-large-en-v1.5`. BGE models pool via the `[CLS]` token by default
  (`1_Pooling/config.json` has `pooling_mode_cls_token: true`), rather than the
  mean-pooling most `sentence-transformers/all-*` models use.
- **Query matching**: `embed_text_local.py`/`dataset.py` reuse
  `experiments/text-searches/trainTextMLP.py`'s exact matching convention:
  `embeddings/all.json` is keyed by per-style-variant render name (e.g.
  `" Really Petshop Italic  Really Petshop Italic"`), `fontQueriesV2.json` by
  base family name (`"Really Petshop"`); match via longest-prefix trie, and
  when a family has multiple matching style variants keep the shortest-named
  one as the canonical "regular" embedding.
- **Train/test split holds out queries, not fonts** (`dataset.splitQueryCache`,
  same convention as `trainTextMLP.py`): each font's own queries are split
  independently (default 80/20), so every font appears in training, and every
  font with more than one query also has held-out queries in test.
- **Diffusion model**: a plain residual MLP (`experiments/diffusion.py:
  DiffusionMLP`) conditioned on a sinusoidal timestep embedding and the raw
  text embedding (no visual information at all besides the noised target) --
  a standard DDPM (linear beta schedule, 1000 steps by default) predicting
  the added noise.
- **The diffusion process operates in a 64-d PCA-whitened subspace, not the
  raw 512-d ambient space** (`dataset.PCAWhitener`, fit on training fonts
  only; `--pcaDim`, default 64). This replaced simple per-dimension
  standardization after diagnosing why early full-scale runs generated
  samples that didn't resemble any real font at all (see "Diagnostic
  findings" below): `embeddings/all.json` has an effective rank of only
  ~57.6/512, so the other ~450 ambient dimensions carry almost no real
  signal, and small per-dimension prediction errors summed across all of
  them can produce a large, essentially random perturbation that swamps the
  real signal. Same idea `utils/search.py`'s `GridFeedbackSearch` already
  uses (centre -> PCA -> whiten) for the same anisotropic embedding space,
  applied here to the diffusion target instead of a search index. 64 was
  chosen to sit a bit above the measured 57.6 effective rank, for headroom.
  `whitener.npz` stores the fit; `evaluate.py`/`diagnose_*.py` project
  generated samples back to the 512-d ambient space (`inverseTransform`)
  before computing cosine similarity against the corpus, so recall@k and
  the diagnostics stay comparable to the earlier full-ambient-space runs.
- **Recall@k evaluation** (`evaluate.py`): for each held-out test query, runs
  `--samplesPerQuery` (default 8) *independent full reverse-diffusion
  processes* from fresh Gaussian noise (no shortcuts), takes the best cosine
  similarity across those samples against every known font's visual
  embedding (fonts are never held out, so the true font is always a
  candidate), and reports whether the true font lands in the top k for
  k in {1, 5, 10, 50, 100}.
- **Verified working** (`experiments/*` as of this commit): smoke-tested
  end-to-end on a 300-font / bge-small subset -- `embed_text_local.py` ->
  `train.py` (3 epochs) -> `evaluate.py` all ran cleanly. Recall@k was 0 at
  that scale/epoch count, as expected (mechanical check only, not a real result).

## Diagnostic findings (first full-scale run, pre-PCA-whitening)

A full run (17,056 matched fonts, 221,888 train / 51,164 test queries, bge-large,
60 epochs, loss 0.433 -> 0.063) got recall@k barely above random chance
(recall@5 0.0004 vs. a 0.000127 random baseline at 39,421 candidates -- roughly
2.4-3.3x chance across k, but tiny in absolute terms). Three diagnostics,
in the order run:

1. **`diagnose_conditioning.py`** (generated-vs-true pairwise similarity
   structure across distinct-font queries): correlation ~0, and generated
   samples averaged only ~0.13 cosine to the corpus-wide mean font vs. real
   fonts' ~0.75 -- with very high pairwise-similarity variance (std ~0.68,
   vs. ~0.044 expected for genuinely random 512-d vectors), the signature of
   a low-dimensional subspace uncorrelated with the real manifold. Identical
   at 50 and 1000 sampling steps, ruling out ancestral-respacing noise as
   the cause.
2. **`diagnose_text_ablation.py`** (denoising loss with real vs. shuffled
   text, holding x0/noise/t fixed, by timestep bucket): real text beat
   shuffled text at every noise level (~1-7% relative, peaking mid-schedule)
   -- ruling out full conditioning collapse. The effect is real, just modest.
3. **`diagnose_manifold_realism.py`** (nearest-neighbor cosine of generated
   samples against the full corpus, vs. real fonts' own leave-one-out
   nearest-neighbor baseline): real fonts sit in an extremely tight cluster
   (mean nearest-other-real cosine ~0.96); generated samples' best match
   anywhere in the corpus averaged only ~0.40. Most generated output doesn't
   resemble *any* real font, correct or not.

Read together: conditioning has a real but modest effect, and generation
quality (not landing near the real, very concentrated manifold at all) looks
like the dominant problem -- consistent with small per-dimension errors,
replicated across the ~450 near-degenerate ambient dimensions
`embeddings/all.json` actually has (see effective dimension below),
dominating the output direction. Not yet re-run against the PCA-whitened
version above to confirm the fix.

## Effective dimension of the visual embedding space

`embeddings/font_vit_embeddings.json` (raw pretrained ViT CLS, mean-pooled over
a-z, ~39.5k fonts) has an effective rank of **~9.5 out of 512** nominal dims
(`effective_dimension.py`) -- true even for individual, unpooled, per-letter
embeddings before any pooling happens (`character_variance.py` rules out
pooling as the cause). `embeddings/all.json` (the contrastively-finetuned
`ViTEmbedder` space) has an effective rank of **~57.6 out of 512** -- about
6x higher, consistent with contrastive/InfoNCE training being what actually
spreads font-identity information across the embedding space; the raw
pretrain objective (reconstruction + character classification) never
incentivizes that. Still fairly anisotropic in absolute terms (~11% of
nominal dims used) -- CLAUDE.md's own "All-but-the-Top" suggestion is a
reasonable inference-time fix worth trying independent of this experiment.

Both embedding files also have the same normalization bug (`pooling_bug_check.py`):
`utils/embeddings.generateEmbeddings` normalizes each per-letter embedding
before mean-pooling but never renormalizes the pooled result, so stored font
vectors don't sit exactly on the unit hypersphere (norms observed: 0.70-1.00
for font_vit_embeddings.json, 0.72-1.00 for all.json) and cosine similarity
against them isn't exactly the average pairwise letter-cosine it's meant to
approximate. Not yet fixed upstream as of this commit.

## Interactive branching search (decision points)

Goal: let a user narrow to a font with a text query plus a handful of
clicks, instead of typing a precise prompt (R@1~=0% above) or doing many
rounds of undifferentiated feedback (`GridFeedbackSearch`'s per-CLAUDE.md
top-m rounds). Idea: run several reverse-diffusion trajectories from the
same text query as a batch of "particles" sharing one conditioning
vector; most of the time they stay unimodal and converge to one answer
with no user input at all, but when they resolve into genuinely distinct
directions, pause, show ~10 representative fonts per direction, let the
user pick one, then keep going with only that branch. Repeat (capped at
5 decisions) until fully denoised.

```
experiments/diffusion-searches/branching.py    # portable primitives: ParticleBatch, advanceTo,
                                                #   findDecisionPoint, previewFonts, pruneAndResample
experiments/diffusion-searches/corpus.py       # FontCorpus (whitened-space ranking) +
                                                #   HierarchicalClusterIndex (the corpus tree)
experiments/diffusion-searches/oracle.py       # simulated user: knows the true target, picks the best
                                                #   branch each round, with an injectable error probability
experiments/diffusion-searches/evaluate_branching.py  # recall@k / decision-count sweep over noiseProb
```

An early version of this (flat, single-level region partition; only
clustering the raw diffusion particles) measured recall@10 ~5-10%. That
number turned out to be an artifact of two real, fixable bugs in the
SEARCH PROTOCOL, not a ceiling imposed by the model or the embedding
space -- see "Three real bugs, all caught by direct measurement" below.
Fixing them raised recall@10 to ~20% (noiseProb=0, n=50) -- roughly 3x
the single-shot baseline's 6% at the same particle budget, and ~3.3x
this search's own earlier number, using the exact same trained checkpoint
throughout, no retraining.

### The corpus is not actually the bottleneck

A user pushback mid-investigation was the right call: "the embeddings
have effective dimension ~57 -- that alone should be enough to identify
each of 39,421 fonts, why is nearest-neighbor cosine ~0.96?" Checked
directly, not assumed:

- **The embedding space's raw information content is not the problem.**
  log2(39,421) ~= 15.3 bits; an effective rank of ~57 real dimensions is
  vastly more than that. Comparing four different similarity metrics
  (ambient L2-normalized cosine -- the pre-existing convention; the raw
  dot product of the un-normalized pooled vectors, which is what the
  known pooling bug (see above) should actually be measuring; whitened-
  space Euclidean; whitened-space cosine) against the SAME generated
  samples gave near-identical single-shot recall@k across all four
  (recall@100 0.03-0.07-ish on a small sample) -- the ranking metric was
  not hiding a large amount of recoverable accuracy.
- **Why nearest-neighbor cosine is nonetheless ~0.96, checked directly**:
  mean-centering the corpus and re-normalizing only drops it to ~0.92 (a
  real but modest anisotropy effect -- the top 5 PCA components explain
  under 20% of variance, so this isn't one dominant "generic" direction
  swamping everything, per CLAUDE.md's "All-but-the-Top" hypothesis).
  Correcting for the documented pooling-normalization bug (comparing raw
  dot products of the un-renormalized per-font mean vectors -- the
  quantity that actually equals the true mean pairwise letter-cosine)
  drops it further, to ~0.85, and reorders the actual nearest-neighbor
  identity in about half of cases. So the ~0.96 figure is a mix of a
  real, measurable metric inflation (~0.96 -> ~0.85 corrected) AND
  genuine corpus density (many fonts in this internet-scraped corpus
  really are near-duplicates/clones/derivatives of each other) -- both
  real, neither one "the embeddings are fundamentally indistinguishable."
- **Direct proof the structure is exploitable**: navigating a
  hierarchical k-means tree of the corpus (branching factor 10, depth 5,
  built once on the whitened embeddings) with an OMNISCIENT oracle --
  no diffusion model involved at all, just always stepping into
  whichever child truly contains a known target -- reached a median
  final group of **4 fonts** within 5 rounds, across 200 sampled targets.
  Implied recall@10 ~97%. The 57 real dimensions are more than adequate;
  what was missing was a search PROTOCOL that could actually address
  that many leaves in 5 rounds.

### Three real bugs, all caught by direct measurement, in order

1. **A flat corpus partition caps addressable resolution regardless of
   round count.** The first design (`corpus.ClusterIndex`, superseded)
   split the corpus into a single k-way partition (e.g. 16 regions) and
   reused the SAME global centroids every round. Once particles settled
   into one region, re-checking against the same centroids couldn't
   narrow any further within it -- recall@10 measured ~10% this way.
   Fix: **`corpus.HierarchicalClusterIndex`**, a real tree (branching
   factor `--branchingFactor`, default 10; depth `--treeDepth`, default
   5) built once offline (~90s for the full 39,421-font corpus on this
   machine; `--treeCache` saves/loads it so repeated eval runs don't
   re-pay that cost). branching^depth = 100,000, comfortably above the
   corpus size -- confirmed exploitable by the omniscient-oracle result
   above.
2. **The decision logic only fired on a genuine tie, and silently did
   nothing the rest of the time.** `assignToChildren`/`findDecisionPoint`
   originally required >=2 children to each clear `--minClusterSize`
   before triggering ANYTHING. The much more common case -- particles
   clearly favoring ONE child with no real second contender -- was
   treated identically to "no signal at all," so sessions just sat at
   the root, never descending. Measured effect: median final node size
   ~100 (barely narrower than the whole corpus), vs. the tree's own
   omniscient-oracle ceiling of ~4. Fix: `assignToChildren` now returns
   a 3-way result -- 0 qualifying children (keep advancing), exactly 1
   (silently auto-descend, no ambiguity to ask about), or >=2 (a genuine
   decision -- ask). This alone brought median final node size down to
   ~6-7, right in line with the omniscient ceiling.
3. **Auto-descending on "confidence" doesn't track correctness, so it
   quietly overrides the very mechanism meant to correct it.** A direct
   measurement (single-shot samples, 60 held-out queries, checked
   against the tree's real top-level membership) found the model's
   majority-vote top-level child choice matches the TRUE target's child
   only 46.7% of the time (vs. 10% random chance at branching factor
   10 -- a real, 4.7x signal, but far from reliable) -- yet being in the
   TOP-3 most-favored children hits 86.7% of the time (vs. 30% chance).
   A model that's "confidently wrong" concentrates particles into one
   child just as readily as a model that's confidently right, so a HIGH
   `--minClusterSize` (a high bar for "is this ambiguous") makes the
   auto-descend path fire on the model's raw (unreliable) top pick far
   more often than it hands control to the oracle -- which, unlike the
   model, is actually reliable when consulted (it scores candidates
   against real acceptance-set similarity, not the model's own belief).
   Fix: a much LOWER `--minClusterSize` (2, vs. an initial 8/40=20%) and
   higher `--persistFor` (4) so weaker secondary contenders register as
   real candidates, biasing the system toward ASKING (using its full
   `--maxRounds` budget) rather than trusting the model's unsupervised
   top pick. Swept minClusterSize in {1,2,3,4,8} x persistFor in {3,4,5}
   on 40 held-out queries at noiseProb=0: minClusterSize 2-3 with
   persistFor 3-4 measured best (recall@10 roughly tripled, recall@50/
   100 nearly doubled, vs. minClusterSize=8) -- minClusterSize=1 was
   tried and found WORSE despite being even more permissive: with nearly
   every child registering >=1 particle, the qualifying SET rarely
   stayed identical across `--persistFor` consecutive checks (too much
   single-particle noise), so sessions often gave up on descending
   entirely rather than committing to a wrong turn -- decent recall@100
   (particles stayed spread over a wide, still-plausible region) but
   worse recall@10 than the tuned middle ground.

### `--maxRounds` (cap decisions at 5)

Only a genuine decision (>=2 real candidates, oracle consulted) counts
against `--maxRounds` (default 5) -- an unambiguous auto-descend isn't a
choice the user had to make, so it's free. Once `--maxRounds` asks are
used up but the tree is still ambiguous, the session silently follows
whichever child the model itself supports most, rather than stalling at
a wide node. In practice sessions use nearly the full budget when tuned
to favor asking (~3.8-4.25 of 5, at n=40-50): mostly because that's
exactly the point -- consult the reliable oracle whenever there's a
real choice, rather than trust the model's own guess.

### Success metric

Recall@k here uses an ACCEPTANCE SET, not the single exact font -- a
session succeeds at k if any of the target's own top-10 nearest
neighbors in the whitened space (`--acceptanceSize`, matching
`GridFeedbackSearch`'s simulated-user convention per CLAUDE.md, not
`evaluate.py`'s stricter single-font metric) lands in the top k of the
final best-of-N ranking (also computed directly in the whitened space --
no ambient round-trip, see corpus.py). At this corpus's density, landing
on a near-identical font is a legitimate product success. A second,
even more direct metric is also reported: whether the search's own FINAL
narrowed group (now typically ~5-10 fonts, not a 39,421-font re-ranking)
actually contains an acceptance-set font -- this is what a real product
would show the user, with no further re-ranking step.

### Recoverability from a wrong decision: hedge and belief, tested and mostly rejected

A hard-pruned tree descent commits 100% of the particle mass to the
chosen branch every round -- an irreversible elimination of everything
else, decided on necessarily imperfect signal (a real user's judgment,
or this model's own ~47% top-1 root accuracy). That's a real, structural
risk, worth testing rather than assuming either way:

1. **A global soft belief (`belief.py`, unused by the default pipeline):**
   an L2-logistic-regression belief refit every round from each round's
   shown-vs-chosen fonts, matching `GridFeedbackSearch`'s own approach in
   `utils/search.py`. Measured **worse than doing nothing**: ~0-1%
   recall@10 on its own, and diluted the working hard-path ranking when
   blended with it (`combined`), especially at higher noise. Diagnosis: a
   session collects far fewer, far more geographically-narrow labels
   (a handful of rounds along one converging path) than
   `GridFeedbackSearch` ever does, which isn't enough to fit a useful
   *global* linear ranking over a 39,421x64 space -- a genuinely
   different label regime, not a tuning failure.
2. **Pooling rejected-branch completions into the final ranking
   (`oracle.runSession`'s `reserveX`/`candidatePool`, still in the
   pipeline as `hedge`):** pooling against the FULL corpus measured worse
   than the hard path alone -- the same dilution effect this project's
   other search work already knows about (more samples help distractor
   candidates just as readily as the true one, and there are far more
   possible distractors than true positives). Restricting the hedge's
   ranking to only branches actually considered fixed the worst of the
   dilution, but an unconditional version still ballooned the candidate
   pool to ~19,000 fonts on average, because an EARLY rejected branch is
   itself huge (a root-level rejection covers ~4,000 fonts) --
   `hedgeMaxBranchSize` caps which rejected branches are cheap enough to
   hedge on at all (large ones are trusted outright; there's no cheap way
   to hedge against being wrong about which quarter of the corpus to
   explore). Even after both fixes, `hedge` still trails `hard` at every
   noise level up to ~0.5 -- see the chart below for where it stops
   trailing.
3. **Forcing an ask (never silently auto-descending) at large nodes
   specifically** (`forceAskThreshold`) measured almost no effect either
   way, because `minClusterSize` was already low enough (2/40 = 5%) that
   a real second contender at a large node usually already triggers an
   ask on its own -- the case this was meant to catch turned out to be
   rare given the other fix already in place.
4. **Branching factor and particle count sweeps**: branchingFactor=14
   (fewer required rounds, but each wrong pick eliminates a larger
   fraction -- (B-1)/B grows with B) measured clearly worse than 10 at
   every noise level tested. numParticles=80 (vs. 40) showed no clear
   improvement at 2x the compute.

### Empirical results (120 held-out queries, 40 particles/round, `branchingFactor=10`,
`treeDepth=5`, `minClusterSize=2`, `persistFor=4`, `maxRounds=5`, 50/1000 reverse-diffusion steps)

Full noiseProb sweep, `hard` vs `hedge`, plotted against the single-shot
(no branching) baseline: **[Recall vs. Oracle Reliability](https://claude.ai/code/artifact/37cece46-ad65-45ec-aab2-f18008541e46)**
(raw numbers in `experiments/diffusion-searches/branching_results_final.json`).

Single-shot baseline (no branching, same particle budget), n=120:
recall@1=0.00, recall@5=0.025, recall@10=0.033, recall@50=0.125, recall@100=0.20.

| noiseProb | hard recall@10 | hedge recall@10 | hard recall@100 | hedge recall@100 |
|---|---|---|---|---|
| 0.0 | 0.200 | 0.117 | 0.450 | 0.325 |
| 0.2 | 0.167 | 0.142 | 0.350 | 0.292 |
| 0.5 | 0.083 | 0.058 | 0.233 | 0.192 |
| 0.7 | 0.050 | 0.025 | 0.158 | 0.192 |
| 1.0 | 0.050 | 0.050 | 0.158 | 0.208 |

Two things are worth taking away, both directly visible in the linked
chart, not just asserted:

- **The degradation is real, clean, and monotonic** (not just "roughly,"
  at n=120): recall@10 (hard) falls 0.20 -> 0.05 as noiseProb goes 0 ->
  1, and recall@100 falls 0.45 -> 0.16, essentially converging to the
  no-branching baseline once the oracle is no better than a coin flip --
  exactly the sanity check a working noise model should pass.
- **`hedge` overtakes `hard` at recall@100 specifically at noiseProb >=
  0.7** (0.19-0.21 vs. 0.16) -- the regime where most or all decisions
  are coin flips. This is the concrete, measured version of the
  fragility concern that motivated testing hedge/belief in the first
  place: an irreversible per-round elimination is riskiest exactly when
  the decision-maker (oracle or real user) is unreliable, and that's
  where keeping a reserve stops being a net cost and starts paying for
  itself. It doesn't win by enough, or soon enough (noiseProb < 0.7), to
  justify as the *default*, but it's a real, data-backed argument for
  widening the reserve specifically when there's reason to distrust the
  current user's picks -- not a blanket policy either way.

**Remaining gap to the tree's own ~97% omniscient-oracle ceiling** (at
noiseProb=0) is attributable to the diffusion model's own per-level
accuracy (measured ~47% top-1 / ~87% top-3 at the root) -- not the
corpus, not the metric, not the search protocol's addressable resolution
or its decision logic (all real, now-fixed bugs; see above), and, per
this section, not fixable by a smarter recovery-from-error mechanism
either -- hedge/belief were tested, not assumed, and both under-deliver
at the noise levels this checkpoint is actually accurate enough to
operate at.

**FiLM conditioning was tried against that remaining gap, and it made
things worse, not better.** `diffusion.ResidualBlock` supports
`conditioning="film"` (Perez et al. 2018 -- text/timestep modulate the
hidden state's own scale/shift instead of being added alongside it) as
well as the original `"concat"`, selectable via `train.py --conditioning`
/`--checkpointDir`. A `checkpoints/diffusion_film` run (identical
hyperparameters and epoch count -- 60 -- to the deployed checkpoint,
conditioning mechanism only) converged to a HIGHER training loss (0.106
vs. 0.063 final) and measured WORSE branching-search recall@10 at
noiseProb=0 (0.117 vs. 0.20, n=60; recall@100 was roughly comparable,
0.40 vs. 0.42). FiLM's extra per-block parameters (a 2x-wider projection)
evidently didn't earn their keep in the same 60-epoch budget. Not a dead
end necessarily -- training the FiLM run longer, or trying
cross-attention conditioning instead, are both still open -- but "try
FiLM conditioning" is a tested, negative result now, not an untried idea.

**Where the accuracy actually breaks down, per a per-tree-level check**
(both checkpoints, 80 queries, following each query's TRUE path
regardless of what the model would have picked -- isolating "how good is
the model's judgment at this depth" from "did an earlier wrong turn even
get us here"): top1/top3 child-prediction accuracy is decent at the root
(concat 46/80%, film 51/83%), drops to its WORST at depth 2-3 (both
~21-25% / ~46-61%), then recovers at depth 4 (~40%/~72-78%, both). FiLM
and concat are tied at every individual depth -- no hidden win the
aggregate number was masking. The dip is concentrated in the MIDDLE of
the tree specifically: plausibly root-level splits are broad, easy-to-
separate macro-styles, and near-leaf splits are among near-duplicate
fonts (easy because it barely matters which one), leaving the FINE-but-
not-yet-trivial middle tiers as where conditioning strength is actually
being tested, and failing most. A concrete target for whoever picks this
up next -- e.g. tree restructuring or targeted training around depth 2-3
discrimination -- rather than another blanket architecture swap.

## Compute note

Font/ViT-embedding generation and the CPU-only DDPM training loop are fine on
a modest sandbox, but text-encoding tens/hundreds of thousands of queries with
a large sentence-transformer model (e.g. bge-large) is not -- run
`embed_text_local.py` locally (ideally with a GPU) rather than in a CPU-only
sandbox; a full bge-large pass over ~380k queries measured at ~3s/batch
(batch 64) there, i.e. hours.
