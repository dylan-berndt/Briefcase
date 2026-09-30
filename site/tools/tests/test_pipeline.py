import json
import os
import subprocess
import sys
import io

import numpy as np
import pytest
import torch
from PIL import Image

import taggers
from bundle import Bundle, BundleError
from bundleWriter import writeBundle
from assembleBundle import assemble
from scoreFonts import renderGlyphs, scoreFonts
from synthFonts import makeFont, LOWER, UPPER, DIGITS
from tinyRun import makeRun, VOCAB
from tagsearch import TagIndex
from render import myfontsStyleFromFont

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)


# ------------------------------------------------------------------ rendering

def test_glyphs_are_rendered_the_way_the_model_was_trained(tmp_path):
    path = makeFont(tmp_path / "a.ttf", LOWER)
    glyphs = renderGlyphs((path, 32))
    assert glyphs.shape == (26, 48, 48) and glyphs.dtype == np.uint8
    assert glyphs.max() == 255 and glyphs.min() == 0
    reference = np.round(myfontsStyleFromFont(path, "q") * 255).astype(np.uint8)
    assert (glyphs[LOWER.index("q")] == reference).all()


def test_glyph_size_follows_the_model(tmp_path):
    path = makeFont(tmp_path / "a.ttf", LOWER)
    assert renderGlyphs((path, 48)).shape == (26, 72, 72)


def test_fonts_that_cannot_render_give_none(tmp_path):
    assert renderGlyphs((str(tmp_path / "missing.ttf"), 32)) is None
    assert renderGlyphs((makeFont(tmp_path / "partial.ttf", LOWER[:-1]), 32)) is None  # no z
    bad = tmp_path / "bad.ttf"
    bad.write_bytes(b"nope")
    assert renderGlyphs((str(bad), 32)) is None


# ------------------------------------------------------------------ the finetuneTags adapter

def test_adapter_reproduces_the_training_model(tmp_path):
    run, model = makeRun(tmp_path / "run")
    tagger = taggers.loadTagger("finetuneTags", run, "cpu")
    assert tagger.vocab == VOCAB and tagger.fontSize == 32

    glyphs = np.stack([renderGlyphs((makeFont(tmp_path / f"{i}.ttf", LOWER, width=200 + 100 * i), 32)) for i in range(3)])
    out = tagger.logits(glyphs)
    assert out.shape == (3, len(VOCAB)) and np.isfinite(out).all()
    model.eval()
    with torch.no_grad():
        expected = model(torch.tensor(glyphs).float() / 255.0).numpy()
    assert np.allclose(out, expected, atol=1e-5)
    assert not np.allclose(out[0], out[2])  # different fonts, different tags


def test_adapter_reads_the_glyph_size_from_the_run(tmp_path):
    run, _ = makeRun(tmp_path / "run", fontSize=48)
    assert taggers.loadTagger("finetuneTags", run, "cpu").fontSize == 48


def test_a_directory_of_runs_picks_the_newest_finished_non_smoke_run(tmp_path):
    makeRun(tmp_path / "2026-09-01 10-00", seed=1)
    makeRun(tmp_path / "2026-09-28 19-25", seed=2)
    makeRun(tmp_path / "2026-09-30 08-00", seed=3, smoke=True)  # a --maxFonts smoke test
    os.makedirs(tmp_path / "2026-10-01 00-00")                   # crashed before saving anything
    assert taggers.newestRun(str(tmp_path), taggers.FineTuneTags.REQUIRED).endswith("2026-09-28 19-25")
    tagger = taggers.loadTagger("finetuneTags", str(tmp_path), "cpu")
    assert tagger.run.endswith("2026-09-28 19-25")


def test_no_finished_run_is_an_error(tmp_path):
    os.makedirs(tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        taggers.loadTagger("finetuneTags", str(tmp_path), "cpu")


def test_unknown_adapter_lists_the_choices(tmp_path):
    with pytest.raises(ValueError, match="finetuneTags"):
        taggers.loadTagger("nope", str(tmp_path), "cpu")


# ------------------------------------------------------------------ font file -> bundle -> search

class InkTagger:
    """Stub model: the more of the glyph box is filled, the bolder. Lets the test know the right answer."""
    vocab = ["bold", "thin", "serif"]
    fontSize = 32

    def __init__(self, midpoint):
        self.midpoint = midpoint

    def logits(self, glyphs):
        ink = glyphs.astype(np.float32).mean(axis=(1, 2, 3)) / 255.0
        bold = 80 * (ink - self.midpoint)
        return np.stack([bold, -bold, np.full_like(bold, -6.0)], axis=1)


@pytest.fixture
def shapes(tmp_path):
    fonts = []
    for name, width in [("Heavy One", 900), ("Thin One", 70), ("Heavy Two", 800), ("Thin Two", 90)]:
        path = makeFont(tmp_path / f"{name}.ttf", LOWER + UPPER + DIGITS + " ", width=width, height=900, family=name)
        fonts.append({"key": f"google:{name}", "name": name, "source": "google", "path": path,
                      "url": f"https://fonts.google.com/specimen/{name.replace(' ', '+')}", "creator": None})
    return fonts


def test_font_files_to_ranked_results(tmp_path, shapes):
    inks = [renderGlyphs((f["path"], 32)).mean() / 255 for f in shapes]
    assert min(inks[::2]) > max(inks[1::2])  # the synthetic heavy fonts really are heavier
    tagger = InkTagger(midpoint=(min(inks[::2]) + max(inks[1::2])) / 2)

    keys, logits, failed = scoreFonts(shapes, tagger, batchSize=3, workers=2)
    assert keys == [f["key"] for f in shapes] and logits.shape == (4, 3) and failed == 0

    scores = {"keys": np.array(keys), "logits": logits.astype(np.float16),
              "vocab": json.dumps(tagger.vocab), "meta": json.dumps({"adapter": "stub"})}
    manifest = assemble(shapes, scores, str(tmp_path / "bundle"), workers=2)
    assert manifest["numFonts"] == 4 and manifest["numTags"] == 3 and manifest["model"] == {"adapter": "stub"}
    assert "path" not in json.loads((tmp_path / "bundle" / "fonts.json").read_text())[0]  # build paths do not ship

    index = TagIndex(Bundle(str(tmp_path / "bundle"), verify=True))
    top = lambda q: {index.bundle.fonts[i]["name"] for i in index.search(q)[0][:2]}  # noqa: E731
    assert top("bold") == {"Heavy One", "Heavy Two"}
    assert top("thin") == {"Thin One", "Thin Two"}
    assert top("not bold") == {"Thin One", "Thin Two"}
    assert top("bold not thin") == {"Heavy One", "Heavy Two"}

    specimen = Image.open(io.BytesIO(index.bundle.specimen(0)))
    assert specimen.format == "WEBP"


def test_fonts_that_cannot_be_rendered_or_scored_are_dropped_not_fatal(tmp_path, shapes):
    bad = tmp_path / "bad.ttf"
    bad.write_bytes(b"nope")
    fonts = shapes + [{"key": "google:Bad", "name": "Bad", "source": "google", "path": str(bad),
                       "url": "https://x", "creator": None}]
    keys, logits, failed = scoreFonts(fonts, InkTagger(0.3), batchSize=2, workers=2)
    assert failed == 1 and "google:Bad" not in keys and len(keys) == 4

    # a font that scored but has no specimen characters is dropped by assemble
    noChars = makeFont(tmp_path / "accents.ttf", "éü")
    extra = {"key": "google:Accents", "name": "Accents", "source": "google", "path": noChars, "url": "https://x", "creator": None}
    scores = {"keys": np.array(keys + ["google:Accents"]),
              "logits": np.concatenate([logits, logits[:1]]).astype(np.float16),
              "vocab": json.dumps(InkTagger.vocab), "meta": "{}"}
    manifest = assemble(shapes + [extra], scores, str(tmp_path / "bundle"), workers=2)
    assert manifest["numFonts"] == 4


# ------------------------------------------------------------------ bundle writer

def test_writer_rejects_inconsistent_input(tmp_path):
    font = {"key": "a", "name": "A", "source": "google", "url": "https://x", "creator": None}
    with pytest.raises(ValueError, match="do not match"):
        writeBundle(str(tmp_path / "b"), [font], ["t1", "t2"], np.zeros((1, 3)), [b"x"])
    with pytest.raises(ValueError, match="one specimen"):
        writeBundle(str(tmp_path / "b"), [font], ["t1"], np.zeros((1, 1)), [])
    with pytest.raises(ValueError, match="unique"):
        writeBundle(str(tmp_path / "b"), [font, font], ["t1"], np.zeros((2, 1)), [b"x", b"y"])


def test_writer_clips_huge_logits_into_float16_range(tmp_path):
    font = {"key": "a", "name": "A", "source": "google", "url": "https://x", "creator": None}
    writeBundle(str(tmp_path / "b"), [font], ["t"], np.array([[1e9]]), [b"x"])
    logits = np.load(tmp_path / "b" / "logits.npy")
    assert np.isfinite(logits).all() and logits[0, 0] == 30


# ------------------------------------------------------------------ the real command line, end to end

def run(*args, cwd):
    result = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_command_line_pipeline_builds_a_bundle_the_server_accepts(tmp_path):
    import pandas as pd
    google = tmp_path / "google" / "fonts" / "ofl"
    for name, width in [("Heavy", 900), ("Thin", 70)]:
        folder = google / name.lower()
        folder.mkdir(parents=True)
        makeFont(folder / f"{name}-Regular.ttf", LOWER + UPPER + DIGITS + " ", width=width, height=900, family=name)
        (folder / "METADATA.pb").write_text(f'name: "{name}"\ncategory: "SANS_SERIF"\n')
    (tmp_path / "dafont" / "fonts").mkdir(parents=True)
    makeFont(tmp_path / "dafont" / "fonts" / "Wobbly.ttf", LOWER)
    pd.DataFrame({"base_font_name": ["Wobbly"], "filename": ["Wobbly.ttf"], "category": ["Fancy"]}).to_csv(tmp_path / "dafont" / "info.csv", index=False)
    (tmp_path / "dafont" / "list.json").write_text(json.dumps([{"name": "Wobbly", "url": "https://www.dafont.com/wobbly.font", "author": "Ann"}]))
    makeRun(tmp_path / "runs" / "2026-09-28 19-25")

    tool = lambda name: os.path.join(TOOLS, name)  # noqa: E731
    out = run(tool("listFonts.py"), "--google", str(tmp_path / "google" / "fonts"), "--dafont", str(tmp_path / "dafont"),
              "--dafontPageList", str(tmp_path / "dafont" / "list.json"), "--out", str(tmp_path / "build" / "fonts.json"), cwd=tmp_path)
    assert "3 fonts" in out
    out = run(tool("scoreFonts.py"), "--fonts", str(tmp_path / "build" / "fonts.json"), "--model", str(tmp_path / "runs"),
              "--out", str(tmp_path / "build" / "scores.npz"), "--workers", "2", "--device", "cpu", cwd=tmp_path)
    assert "2026-09-28 19-25" in out and "scored 3 fonts" in out
    out = run(tool("assembleBundle.py"), "--fonts", str(tmp_path / "build" / "fonts.json"), "--scores", str(tmp_path / "build" / "scores.npz"),
              "--out", str(tmp_path / "data"), "--workers", "2", cwd=tmp_path)
    assert "3 fonts x 6 tags" in out

    bundle = Bundle(str(tmp_path / "data"), verify=True)
    assert sorted(f["key"] for f in bundle.fonts) == ["dafont:Wobbly", "google:Heavy", "google:Thin"]
    assert bundle.manifest["model"]["path"].endswith("2026-09-28 19-25")

    from app import createApp
    app = createApp({"BUNDLE_DIR": str(tmp_path / "data"), "DATABASE": str(tmp_path / "t.db"), "RATELIMIT_ENABLED": False})
    body = app.test_client().get("/api/font/query?query=bold").json
    assert body["total"] == 3 and {r["name"] for r in body["results"]} == {"Heavy", "Thin", "Wobbly"}
