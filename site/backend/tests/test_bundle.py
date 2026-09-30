import json
import os
import shutil

import numpy as np
import pytest

from bundle import Bundle, BundleError


@pytest.fixture
def copy(fake, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(fake[0], target)
    return target


def test_loads_and_matches_manifest(fake):
    bundle = Bundle(fake[0], verify=True)
    assert len(bundle.fonts) == 300
    assert bundle.logits.shape == (len(bundle.vocab), 300)
    assert bundle.logits.dtype == np.float16
    assert bundle.indexByKey[bundle.fonts[17]["key"]] == 17


def test_specimens_are_webp_slices(fake):
    bundle = Bundle(fake[0])
    for i in (0, 1, 299):
        data = bundle.specimen(i)
        assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    assert bundle.specimen(0) != bundle.specimen(1)


def test_missing_manifest(tmp_path):
    with pytest.raises(BundleError, match="manifest"):
        Bundle(str(tmp_path))


@pytest.mark.parametrize("name", ["vocab.json", "fonts.json", "logits.npy", "specimens.pack"])
def test_missing_file(copy, name):
    os.remove(copy / name)
    with pytest.raises(BundleError, match=name):
        Bundle(str(copy))


def test_git_lfs_pointer_is_rejected(copy):
    (copy / "logits.npy").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
    with pytest.raises(BundleError, match="lfs"):
        Bundle(str(copy))


def test_truncated_pack_is_rejected(copy):
    with open(copy / "specimens.pack", "r+b") as file:
        file.truncate(100)
    with pytest.raises(BundleError, match="bytes"):
        Bundle(str(copy))


def test_checksum_mismatch_only_with_verify(copy):
    with open(copy / "specimens.pack", "r+b") as file:
        file.seek(10)
        file.write(b"\x00\x01\x02")
    Bundle(str(copy))
    with pytest.raises(BundleError, match="checksum"):
        Bundle(str(copy), verify=True)


def test_wrong_format_version(copy):
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["format"] = 99
    (copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="format"):
        Bundle(str(copy))


def test_duplicate_keys(copy):
    fonts = json.loads((copy / "fonts.json").read_text())
    fonts[1]["key"] = fonts[0]["key"]
    (copy / "fonts.json").write_text(json.dumps(fonts))
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["files"]["fonts.json"]["bytes"] = os.path.getsize(copy / "fonts.json")
    (copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="duplicate"):
        Bundle(str(copy))


def test_vocab_fonts_shape_mismatch(copy):
    vocab = json.loads((copy / "vocab.json").read_text())[:-1]
    (copy / "vocab.json").write_text(json.dumps(vocab))
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["files"]["vocab.json"]["bytes"] = os.path.getsize(copy / "vocab.json")
    (copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="expected"):
        Bundle(str(copy))
