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
`STATIC_DIR`, `TAG_VOCABULARY`, `COOKIE_SECURE=0` for plain http, `VERIFY_BUNDLE=1` to check sha256s at startup.

## API

| | |
|---|---|
| `GET /api/font/query?query=&page=1&pageSize=24` | `{results, page, pageSize, total, totalPages, tags, unmatched}`; `pageSize` 1-100; a page past the end is empty. Results carry `rating {average, count, mine}` and the caller's `vote` for this query |
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
