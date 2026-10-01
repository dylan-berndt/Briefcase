# The site

Free-text font search over Google Fonts and the dafonts-free subset of DaFont. A query is parsed into style tags
("elegant script, not too thin" -> `elegant`, `script`, not `thin`), every font is scored on those tags, and the
results are shown as specimen images, 24 to a page, with as many pages as the user cares to read. Each result links
out to its Google Fonts or DaFont page. Logged-in users can mark a result as matching or not matching the search,
rate the font 1-5 stars, and describe it.

No model runs on the server. A tagger scores every font offline; the server only reads the scores.

```
site/tools/   offline: pick the fonts, run the tagger, render specimens, write the bundle   (needs torch, a GPU helps)
site/backend/ Flask + numpy: load the bundle, parse queries, rank, serve specimens, accounts/votes/ratings/descriptions
site/frontend/ React app, built into the image
site/e2e/     browser tests against the built site
```

## The search bundle (`site/backend/data`)

| file | contents |
|---|---|
| `manifest.json` | format version, which model produced the scores, size and sha256 of every other file |
| `vocab.json` | the tagger's tags, in column order |
| `fonts.json` | per font: `key` (`google:<family>` / `dafont:<family>`, what votes and ratings are filed under), `name`, `source`, `url`, `creator`, `specimen` `[offset, length]` |
| `logits.npy` | float16 `[numTags, numFonts]`, the tagger's logits. Tag-major, so a query reads a few contiguous rows |
| `specimens.pack` | every specimen WebP back to back (one LFS object instead of ~20k) |

The server refuses to start on a missing, truncated or mismatched bundle, including a git-lfs pointer left by a
checkout without `git lfs pull`. The Dockerfile runs the same check at build time. The large files are tracked with
git-lfs (`.gitattributes`); `manifest.json` is plain text.

### How a query is scored

`utils/tagVocabulary.py` turns the text into weighted tag groups using the reviewed alias table in
`configs/tagVocabulary.json` (186 canonical tags merging 606 MyFonts tags, with negation). Every other tag the model
predicts is searchable by its own name, so the tagger's whole vocabulary is reachable, not only the reviewed one.

Each font's tag probabilities are normalised to sum to 1 over the whole vocabulary (a semantic multinomial,
Turnbull et al. 2008; it removes the bias towards fonts that score high on everything). A positive group scores
`w * log(mass on the group's tags)`, a negated group `|w| * log(probability of none of them)`, and the terms are
summed. The whole corpus is ranked on every request (about 10 ms for 40k fonts x 1.3k tags) and sliced into pages, so
pages never overlap and there is no depth limit.

The negation term and using the full vocabulary for normalisation are design choices made without real-data
validation; `e23_scoring.py` measured the multinomial for positive multi-tag queries only.

**Words the vocabulary does not know.** A word that matches no phrase as typed matches a single-word phrase with the
same Porter2 stem (the snowballstemmer package's English stemmer, applied to both sides: sketched -> sketch, swirling -> swirls; of
several such phrases, the one sharing the longest prefix). Stems do not reduce comparatives (bolder) or -y adjectives
to their root (slimy is not slim). A word that still matches nothing is looked up, as typed or by stem, in
`configs/wordTags.json`, a table of caption words and the tags they co-occur with, learned from the LLM captions of MyFonts fonts (the captions
were written from each font's real tags). The word becomes one extra group over the union of its tags ("airy" ->
thin or feminine) at half weight, and is reported under `inferred` so the page can show it as a removable chip;
sending the word back in `ignore` drops the guess. Built by `site/tools/buildWordTags.py` (log-odds z-score with an
informative Dirichlet prior, Monroe et al. 2008; `configs/wordTagsExclude.txt` holds reviewed non-style words). On
held-out single-word aliases of the reviewed vocabulary, 58% are in the table and, of those, 77% get the right tag
first and 89% in the top 3. Words that the MyFonts captions never use (greasy, slimy, melting) have no entry.

**Suggested tags.** A word that still matches nothing gets up to five suggested tags from a plain synonym check
(`site/backend/synonyms.py`): spaCy word vectors (`en_core_web_md`) compared with the words the search knows (the
single-word aliases and the caption-table words), keeping the tags of neighbours above 0.55 cosine ("slimy" ~ dirty ->
distressed, grunge). They are returned under `suggested` and do not affect the ranking; the page can offer them as
unselected chips and send a chosen one back in `tags=`. Word vectors put antonyms together ("wet" ~ dry), and the medium
model shares one vector between many rare words, so suggestions are noisy by design. `SYNONYM_MODEL=` (empty) turns
them off; they are also off when spaCy or the model is not installed.

**Chips.** The page shows what the engine understood under the search box. Solid chips are tags in the search (matched
from the typed words, or added from a suggestion, which can be removed again); a red chip is a negated tag. A dashed chip
is a caption-table guess for a word the vocabulary does not know ("airy → thin, feminine"); removing it sends the word in
`ignore=`. Dotted `+` chips are synonym suggestions for words that matched nothing; clicking one sends it in `tags=`. Both
lists are in the page URL (`?q=...&tags=a,b&ignore=word`), so reload, back/forward and shared links keep them, and a new
search starts clean. Words with suggestions are not also listed as "not recognised".

## Building the bundle

Run from the repo root, with the research environment (`requirements.txt`) and the datasets in place.

```bash
# 1. which fonts: Google Fonts + dafonts-free, without dingbats/icon fonts and fonts lacking a-z
python site/tools/listFonts.py --google google/fonts --dafont dafont --dafontPageList dafont/cache/font_list.json

# 2. run the tagger over them (the only step that needs the model)
python site/tools/scoreFonts.py                                  # defaults: finetuneTags, newest finished run
python site/tools/scoreFonts.py --model "checkpoints/retrieval/finetuneTags/2026-09-28 19-25"

# 3. specimens + bundle
python site/tools/assembleBundle.py                              # writes site/backend/data
git add site/backend/data && git commit                          # through git-lfs
```

The caption table is rebuilt separately, when the vocabulary or the tagger's tag list changes (needs the research
environment, `results/fontQueries*.json` and `dataset/taglabel`):

```bash
python site/tools/buildWordTags.py      # writes configs/wordTags.json and prints the held-out evaluation
```

Intermediate files go to `build/` (ignored). Steps 1 and 3 do not depend on the model, so iterating on the model means
rerunning step 2 and then step 3.

**Using a different model.** `scoreFonts.py --adapter NAME --model PATH`. An adapter (`site/tools/taggers.py`) is a
class with `load(path, device)` returning an object with `vocab`, `fontSize` and `logits(glyphs)`, where `glyphs` is
`uint8 [B, 26, H, W]`, lowercase a-z rendered the MyFonts way (the same rendering `finetuneTags.py` trained on). Add it
to `ADAPTERS`. Nothing after scoring knows about the model.

`listFonts.py` assumes the layouts `labels.py`/`names.py` read: `google/fonts/*/*/METADATA.pb` beside the font files,
`dafont/info.csv` (`base_font_name`, `filename`, `category`) with files under `dafont/fonts/`, and a dafonts-free
`cache/font_list.json` giving each family's `name` and a page (`url`/`link`/`href`/`page`) and creator
(`creator`/`author`/`designer`). It reports how many fonts it left out and why; check that DaFont fonts were not all
skipped for lack of a page.

## Running locally

```bash
# a fake bundle in the real format (fonts called "Fake 0001", placeholder specimens)
python site/tools/fakeBundle.py site/backend/data-dev

cd site/frontend && npm ci && npm run build && cd ../backend
pip install -r requirements.txt
SECRET_KEY=dev BUNDLE_DIR=data-dev SQLITE_PATH=/tmp/fontsearch.db COOKIE_SECURE=0 \
    STATIC_DIR=../frontend/build python app.py         # http://localhost:8000
```

Environment: `SECRET_KEY` (required), `SQLITE_PATH` (users, votes, ratings, descriptions), `BUNDLE_DIR`,
`STATIC_DIR`, `TAG_VOCABULARY`, `COOKIE_SECURE=0` for plain http, `VERIFY_BUNDLE=1` to check sha256s at startup,
`SYNONYM_MODEL` (default `en_core_web_md`, empty for no suggested tags).

## API

| | |
|---|---|
| `GET /api/font/query?query=&page=1&pageSize=24&ignore=&tags=` | `{results, page, pageSize, total, totalPages, tags, unmatched, inferred}`; `pageSize` 1-100; a page past the end is empty. `inferred` is `[{word, tags, weight}]` for words matched only through the caption table; `ignore` is a comma-separated list of such words not to infer. `suggested` is `[{word, tags: [{tag, via, similarity}]}]` for words that matched nothing (not used in the ranking); `tags` is a comma-separated list of tags to add to the query, `-name` to exclude one (unknown names come back in `unmatched`). Results carry `rating {average, count, mine}` and the caller's `vote` for this query |
| `GET /api/font/specimen/<i>?v=<bundle version>` | the specimen WebP, cached for a year |
| `POST /api/font/approve` `{fontKey, query, vote}` | does this font answer this query: 1, -1, or 0 to clear. Per user, per query (queries are lower-cased and whitespace-collapsed) |
| `POST /api/font/rate` `{fontKey, rating}` | is this a good font: 1-5, or 0 to clear. Per user, per font |
| `POST /api/font/describe` `{fontKey, description}` | up to 500 characters |
| `POST /api/font/register`, `login`, `logout`; `GET /api/font/me` | accounts; a 1-hour JWT in an HttpOnly cookie |

The three feedback endpoints need a login. Votes, ratings and descriptions live in SQLite (`fontVotes`,
`fontRatings`, `fontDescriptions`) keyed by font key. The tables of the previous schema (`fonts`, `fontsMeta`,
`registry`, `ratings`, `approvals`, `descriptions`) are left in an existing database untouched; only `users` carries
over.

## Map page

The Map tab shows `flower.html` / `blob.html` (Plotly exports) in an iframe. They are generated outside the repo: put them in
`site/frontend/public/` (un-ignored in `.gitignore`, tracked with git-lfs). A missing file is a 404, not the app.

## Tests

```bash
cd site/backend && pip install pytest pillow && pytest          # bundle, tag search, every endpoint
cd site/tools && pytest                                         # font list, rendering, adapter, bundle writer, CLI pipeline (needs the research env)
cd site/frontend && CI=true npm test -- --watchAll=false        # components: search, paging, feedback, login
pytest site/e2e                                                 # real browser, see site/e2e/README.md
```
