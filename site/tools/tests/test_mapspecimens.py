import io
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mapSpecimens import BASELINE, HEIGHT, buildMap, indexFiles, renderName, resolve  # noqa: E402
from synthFonts import DIGITS, LOWER, UPPER, makeFont  # noqa: E402

NAME = LOWER + UPPER + DIGITS + " "


def decode(data):
    return Image.open(io.BytesIO(data)).convert("RGBA")


def inkBox(image):
    return image.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox()


def test_a_name_is_drawn_one_line_tall_in_its_own_font(tmp_path):
    path = makeFont(tmp_path / "a.ttf", NAME, family="Synth Alpha")
    image = decode(renderName(path, "Synth Alpha Regular"))
    assert image.height == HEIGHT
    assert inkBox(image) is not None


def test_every_font_sits_on_the_same_baseline_whatever_its_size(tmp_path):
    wide = decode(renderName(makeFont(tmp_path / "w.ttf", NAME, width=700, height=900), "Wide Font"))
    narrow = decode(renderName(makeFont(tmp_path / "n.ttf", NAME, width=200, height=500), "Narrow Font"))
    assert wide.height == narrow.height == HEIGHT
    assert inkBox(wide)[3] == inkBox(narrow)[3] == BASELINE        # the rectangles stand on the baseline in both
    assert inkBox(wide)[1] < inkBox(narrow)[1]                     # the taller font rises higher above it


def test_a_long_name_is_shrunk_to_fit_not_cut(tmp_path):
    path = makeFont(tmp_path / "w.ttf", NAME, width=700)
    normal = decode(renderName(path, "Short Name"))
    image = decode(renderName(path, "A Very Long Family Name Indeed Semi Bold"))
    assert image.width <= 570
    assert inkBox(image)[2] <= image.width - 4                     # nothing runs off the edge
    assert inkBox(image)[1] > inkBox(normal)[1]                    # it was drawn smaller to fit


def test_a_font_that_cannot_draw_its_name_shows_its_glyphs_and_the_name(tmp_path):
    path = makeFont(tmp_path / "d.ttf", DIGITS, family="Dingbats")
    fallback = decode(renderName(path, "Dingbats Regular"))
    assert fallback.height == HEIGHT and fallback.width > 100
    # the glyphs are bright ink, the name after them is the muted grey
    solid = [pixel[:3] for pixel in fallback.getdata() if pixel[3] > 200]       # WebP shifts colours a little
    assert any(min(c) > 215 for c in solid)
    assert any(all(100 < v < 140 for v in c) for c in solid)


def test_a_font_with_nothing_to_draw_is_left_out(tmp_path):
    assert renderName(makeFont(tmp_path / "e.ttf", " "), "Empty Regular") is None


def files(tmp_path):
    a = makeFont(tmp_path / "Abel-Regular.ttf", NAME, family="Abel")
    b = makeFont(tmp_path / "Abel[wght].ttf", NAME, family="Abel")
    c = makeFont(tmp_path / "Cabin-Regular.ttf", NAME, family="Cabin")
    return a, b, c


def test_labels_find_their_files_by_the_name_pillow_reads(tmp_path):
    a, b, c = files(tmp_path)
    by = indexFiles({"google": str(tmp_path)})
    assert set(by) == {"Abel Regular", "Cabin Regular"}
    assert len(by["Abel Regular"]) == 2 and by["Cabin Regular"] == [("google", c)]


def test_the_regular_file_is_preferred_where_several_share_a_label_and_a_suffix_picks_the_source(tmp_path):
    a, b, c = files(tmp_path)
    by = {"Abel Regular": [("google", b), ("google", a), ("dafont", c)]}
    assert resolve("Abel Regular", {"google", "dafont"}, by)[0] == a
    assert resolve("Abel Regular (dafont)", {"google", "dafont"}, by) == (c, "Abel Regular")
    assert resolve("Abel Regular (google)", {"google"}, by)[0] == a
    assert resolve("Abel Regular (dafont)", {"google"}, by)[0] is None      # that source is not on this map
    assert resolve("Nobody Regular", {"google"}, by) == (None, "Nobody Regular")


def test_a_map_gets_an_index_in_point_order_and_a_file_is_drawn_once(tmp_path):
    a, b, c = files(tmp_path)
    by = indexFiles({"google": str(tmp_path)})
    rendered = {}
    pack, index, shown = buildMap(["Abel Regular", "Nobody Regular", "Cabin Regular", "Abel Regular"], {"google"}, by, rendered)
    assert index["count"] == 4 and shown == 3
    assert index["specimens"][1] is None and index["specimens"][0] == index["specimens"][3]
    (offset, length) = index["specimens"][2]
    assert decode(pack[offset:offset + length]).height == HEIGHT
    assert len(rendered) == 2                                              # two files drawn, not three images
    again, _, _ = buildMap(["Cabin Regular"], {"google"}, by, rendered)    # a second map reuses what is drawn
    assert len(rendered) == 2 and again == pack[offset:offset + length]
