# The site

Free-text font search over Google Fonts and the dafonts-free subset of DaFont. A query is parsed into style tags
("elegant script, not too thin" -> `elegant`, `script`, not `thin`), every font is scored on those tags, and the
results are shown as specimen images, 24 to a page, with as many pages as the user cares to read. Each result links
out to its Google Fonts or DaFont page. Logged-in users can mark a result as matching or not matching the search
(thumbs up/down). The page no longer shows star ratings or the description box, but `/api/font/rate` and
`/api/font/describe` are still there.

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
`configs/tagVocabulary.json` (186 canonical tags merging 606 MyFonts tags). Every other tag the model
predicts is searchable by its own name, so the tagger's whole vocabulary is reachable, not only the reviewed one.

Each font's tag probabilities are normalised to sum to 1 over the whole vocabulary (a semantic multinomial,
Turnbull et al. 2008; it removes the bias towards fonts that score high on everything). A group scores
`w * log(mass on the group's tags)` and the terms are summed. There is no negation: "not thin" yields no tag. The whole corpus is ranked on every request (about 10 ms for 40k fonts x 1.3k tags) and sliced into pages, so
pages never overlap and there is no depth limit.

Using the full vocabulary for normalisation is a design choice made without real-data validation; `e23_scoring.py`
measured the multinomial for positive multi-tag queries only.

**Words the vocabulary does not know.** A word that matches no phrase as typed matches a single-word phrase with the
same Porter2 stem (the snowballstemmer package's English stemmer, applied to both sides: sketched -> sketch, swirling -> swirls; of
several such phrases, the one sharing the longest prefix). Stems do not reduce comparatives (bolder) or -y adjectives
to their root (slimy is not slim). A word that still matches nothing is looked up, as typed or by stem, in
`configs/wordTags.json`, a table of caption words and the tags they co-occur with, learned from the LLM captions of MyFonts fonts (the captions
were written from each font's real tags). The word becomes one extra group over the union of its tags ("airy" ->
thin or feminine) at half weight; `/api/font/tags` reports those tags like any others (weight 0.5). Built by `site/tools/buildWordTags.py` (log-odds z-score with an
informative Dirichlet prior, Monroe et al. 2008; `configs/wordTagsExclude.txt` holds reviewed non-style words). On
held-out single-word aliases of the reviewed vocabulary, 58% are in the table and, of those, 77% get the right tag
first and 89% in the top 3. Words that the MyFonts captions never use (greasy, slimy, melting) have no entry.

**Suggested tags.** A word that still matches nothing gets up to eight suggested tags from a WordNet table
(`site/backend/synonyms.py`, reading `configs/synonymTags.json`). WordNet is a hand-built thesaurus, so no web text and
none of its associations. `site/tools/buildSynonyms.py` builds the table offline: for every adjective and noun the search
does not already understand it takes the words WordNet relates to it (same-sense synonyms, "similar to", "see also"),
checks each against what the search does with a single word (aliases, stems, the caption table) and keeps the tags they
reach. Only the first senses count, weighted by how often WordNet's sense-tagged text uses each (verbs are skipped:
"fancy" the verb is "imagine"), each sense gets one vote per tag, and senses combine as a noisy-or, so several words from
one wrong sense do not outweigh one word from the right one. "wet" gives drip (via drippy), sloppy and steam.
`/api/font/tags` returns them under `suggested`; they are not part of the ranking until the page sends one in `tags=`.
Rebuild when the vocabulary, the caption table or the model's tags change:

    pip install nltk && python -m nltk.downloader wordnet
    python site/tools/buildSynonyms.py --examples wet old fancy    # needs the real site/backend/data/vocab.json (git-lfs)

`SYNONYMS=` (empty) turns suggestions off. Unlike the spaCy vectors this replaced, nothing is loaded into memory but a
small JSON table, and the server image has no spaCy.

**The tag line.** The server only lists tags; the page owns the ticks. On a search the page asks
`GET /api/font/tags?query=` once, and shows the answer as one line of words under the search box, each with a box: ticked
(included) or empty (off). The words are the tags the query matched, the tags guessed for unknown words (ticked,
ordinary tags) and the synonym suggestions (empty). A click flips the box (an empty suggestion ticks on its first click). Every change is page state and sends
the new list to `GET /api/font/query?tags=name:weight,name:weight,...`, which ranks exactly those tags; the weight is the one
`/api/font/tags` reported (0.6 for a loose alias, 0.5 for a guess), the page just passes it along. Nothing about the ticks is
in the URL (`?q=...&page=`) or stored: a reload starts from the query's own tags. Changing a tag goes back to page 1; searching
again, even the same words, starts clean. The list wraps when it does not fit one line. Because a guessed word's tags are
now separate tags, each at half weight, a guess is "all of these" where the old caption-table group was "any of these".

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
`SYNONYMS` (default `auto`: `configs/synonymTags.json`; empty for no suggested tags).

## API

| | |
|---|---|
| `GET /api/font/tags?query=` | the tags a query means: `{tags: [{tag, weight}], suggested: [{tag, via, score}], unmatched}`. `suggested` are WordNet-related tags for words that matched nothing (not part of the search until sent back); `unmatched` are words with neither |
| `GET /api/font/query?query=&tags=&page=1&pageSize=24` | `{results, page, pageSize, total, totalPages, tags}`; `pageSize` 1-100; a page past the end is empty. With `tags` (comma-separated `name`, each optionally `:weight` above 0 and up to 1; `-name` is a 400, there is no exclusion) the fonts are ranked on exactly that list and `query` is only the label votes are filed under; unknown names are skipped, a bad entry is a 400, an empty list gives no results. Without `tags` the query text is parsed (guesses included). Results carry `rating {average, count, mine}` and the caller's `vote` for this query and these tags |
| `GET /api/font/specimen/<i>?v=<bundle version>` | the specimen WebP, cached for a year |
| `POST /api/font/approve` `{fontKey, query, tags?, vote}` | does this font answer this query: 1, -1, or 0 to clear. Per user, per query and tag list: the query is lower-cased and whitespace-collapsed, `tags` is the list the results were ranked on (same format as `/api/font/query`, stored sorted as `bold:1,serif:0.5`; without it, the tags the query text means). The same font and query under other ticks is a separate vote |
| `POST /api/font/rate` `{fontKey, rating}` | is this a good font: 1-5, or 0 to clear. Per user, per font (not used by the page at the moment) |
| `POST /api/font/describe` `{fontKey, description}` | up to 500 characters |
| `POST /api/font/register`, `login`, `logout`; `GET /api/font/me` | accounts; a 1-hour JWT in an HttpOnly cookie |

The three feedback endpoints need a login. Votes, ratings and descriptions live in SQLite (`fontVotes`,
`fontRatings`, `fontDescriptions`) keyed by font key. A `fontVotes` row is `userID, fontKey, query, tags, vote, created`;
databases from before `tags` existed are rebuilt on startup and their votes keep `tags = ''` (unknown). The tables of
the previous schema (`fonts`, `fontsMeta`, `registry`, `ratings`, `approvals`, `descriptions`) are left in an existing
database untouched; only `users` carries over.

## About page

`site/frontend/src/about/about.md` is the page: edit it and rebuild, nothing else changes. It is rendered with `marked`
in the site font (Patua One). A first-line `#` heading is the title; the `##` and `###` headings become a table of
contents under it (left out when there are fewer than two). The file is built into the bundle as a static asset and
fetched on load, so it is trusted content: raw HTML in it is passed through.

## Pages and addresses

The pages are routes (React Router, `BrowserRouter` in `src/index.js`, routes in `src/App.js`): `/` search, `/map`, `/about`;
any other address redirects to `/`. The header entries are real links. Flask already serves `index.html` for every path
without a file extension, so a direct visit or a reload of any of them works. The search keeps its own `?q=&page=`
handling; Home does nothing while already on the search page, so the results stay.

## Phones

Below 700px wide the dark column is the whole width of the screen (`--ui-width` in `App.css`), the header buttons tighten
and grow to a finger's height, fields are 16px (smaller makes a phone's browser zoom in on tap), and the About page is one
column with its contents first, in a box capped at 35% of the screen (below 900px).

The background shader becomes a slice pinned to the bottom of the screen and laid over the page (`--shader-band`:
10.66% of the screen's small height, so 90px on a 390x844 phone and 68px on 360x640; taps go through it). It is the same
canvas, shrunk to the slice, so it renders only the slice. A page ends above it, not under it. The column's bottom edge is the top of the slice, with an 8px border there in the column's own colour
(`--column-edge`, `.Shader::before`): a result scrolling under the slice stops short of the shader at every scroll
position, instead of being cut off against it. Its drop shadow (4vmin, black, as on a desktop) falls onto the slice;
both are drawn on the slice (`.Shader::before` and `::after`) because it is fixed and laid over the page, which would hide
a box-shadow on the column. On a desktop the shader is
still the whole screen behind the column.

The pattern is laid out per canvas pixel and the canvas renders at 1/9 resolution, so blocks and noise cells are the same
size in CSS pixels on every screen; it used to scale them by the element's width, which gave a phone a few huge blocks and
almost no pattern. `e2e/test_site.py::test_pages_fit_a_phone_or_tablet_screen` and
`test_phones_have_a_slice_of_the_background_pinned_to_the_bottom` check sideways scrolling, header sizes, the About layout
and the slice at several phone sizes.

## Sitemap

`site/frontend/public/sitemap.xml` lists the three pages (`/`, `/map`, `/about`) at `https://font-search.com`, and
`robots.txt` points to it. Search results (`/?q=...`) are not listed: there is no end to them. The files are copied into
the build and served as static files. If a page is added, add it to the sitemap (`e2e/test_site.py` checks every listed
address is served as the app).

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
