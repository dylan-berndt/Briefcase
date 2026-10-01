import io

import numpy as np
from PIL import Image

from specimens import renderSpecimen, WIDTH, HEIGHT
from synthFonts import makeFont, LOWER, UPPER, DIGITS


def decode(data):
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def test_specimen_is_a_webp_with_ink_on_transparent(tmp_path):
    path = makeFont(tmp_path / "a.ttf", LOWER + UPPER + DIGITS + " ")
    data = renderSpecimen(path)
    image = decode(data)
    assert image.format == "WEBP" and image.size == (WIDTH, HEIGHT) and image.mode == "RGBA"
    alpha = np.asarray(image)[:, :, 3]
    assert 0 < (alpha > 0).mean() < 0.8  # some ink, not a filled block
    assert alpha[0, 0] == 0  # transparent background
    assert len(data) < 20_000


def test_lines_the_font_cannot_draw_are_left_out_not_drawn_as_boxes(tmp_path):
    lower = decode(renderSpecimen(makeFont(tmp_path / "l.ttf", LOWER)))
    full = decode(renderSpecimen(makeFont(tmp_path / "f.ttf", LOWER + UPPER + DIGITS + " ")))
    inkLower = (np.asarray(lower)[:, :, 3] > 0).sum()
    inkFull = (np.asarray(full)[:, :, 3] > 0).sum()
    assert 0 < inkLower < inkFull  # the second line only exists when the font has those characters


def test_font_with_no_sample_characters_gives_none(tmp_path):
    assert renderSpecimen(makeFont(tmp_path / "x.ttf", "éü")) is None


def test_unreadable_file_gives_none(tmp_path):
    bad = tmp_path / "bad.ttf"
    bad.write_bytes(b"not a font")
    assert renderSpecimen(str(bad)) is None


def test_long_lines_are_shrunk_to_fit(tmp_path):
    wide = makeFont(tmp_path / "wide.ttf", LOWER + UPPER + DIGITS + " ", width=1800, height=700)
    alpha = np.asarray(decode(renderSpecimen(wide)))[:, :, 3]
    columns = np.where((alpha > 0).any(axis=0))[0]
    assert columns.min() >= 8 and columns.max() <= WIDTH - 8  # nothing cut off at the edges


def test_a_font_that_cannot_be_scaled_is_left_out_not_fatal(tmp_path, monkeypatch):
    # fixed-size bitmap fonts (colour emoji, PCF) raise "invalid pixel size" from FreeType at any other size
    import specimens
    path = makeFont(tmp_path / "bitmap.ttf", LOWER + UPPER + DIGITS + " ")
    def refuse(*args, **kwargs):
        raise OSError("invalid pixel size")
    monkeypatch.setattr(specimens.ImageFont, "truetype", refuse)
    assert renderSpecimen(path) is None
