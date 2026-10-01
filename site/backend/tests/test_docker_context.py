"""The image is built from a context that .dockerignore whitelists file by file, so a COPY of anything not listed there
fails the deploy ("not found") even though the file is in the repo. This checks every COPY source in the Dockerfile
against the ignore rules, which cannot be tried here without a Docker daemon."""

import glob
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def toRegex(pattern):
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out + r"\Z")


def ignored(path, patterns):
    """Docker's rule: the last pattern that matches the path (or a directory above it) decides."""
    parts = path.split("/")
    prefixes = ["/".join(parts[:n]) for n in range(1, len(parts) + 1)]
    result = False
    for pattern in patterns:
        negated = pattern.startswith("!")
        regex = toRegex(pattern.lstrip("!").strip("/"))
        if any(regex.match(prefix) for prefix in prefixes):
            result = not negated
    return result


def readPatterns():
    with open(os.path.join(ROOT, ".dockerignore")) as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


def copySources():
    sources = []
    with open(os.path.join(ROOT, "Dockerfile")) as file:
        for line in file:
            parts = line.split()
            if parts and parts[0] == "COPY" and not any(p.startswith("--from") for p in parts):
                sources += [p.rstrip("/") for p in parts[1:-1] if not p.startswith("--")]
    return sources


def test_every_copied_file_exists_and_is_in_the_build_context():
    sources = copySources()
    assert "configs/wordTags.json" in sources and "site/backend" in sources   # the parser is reading the Dockerfile
    patterns = readPatterns()
    for source in sources:
        found = [os.path.relpath(p, ROOT).replace(os.sep, "/") for p in glob.glob(os.path.join(ROOT, source))]
        assert found, f"{source} is copied but is not in the repo"
        for path in found:
            assert not ignored(path, patterns), f"{path} is copied by the Dockerfile but .dockerignore leaves it out"


@pytest.mark.parametrize("path", ["site/backend/tests/test_api.py", "site/frontend/node_modules/x/y.js",
                                  "site/backend/data-dev/manifest.json", "configs/vit.json", "checkpoints/pretrain/best/config.json",
                                  "site/backend/__pycache__/app.cpython-311.pyc"])
def test_what_should_stay_out_stays_out(path):
    assert ignored(path, readPatterns())


def test_the_matcher_would_have_caught_the_missing_file():
    # .dockerignore as it was when the deploy failed
    old = ["*", "!site/backend", "!site/frontend", "!utils/tagVocabulary.py", "!configs/tagVocabulary.json"]
    assert ignored("configs/wordTags.json", old)
    assert not ignored("configs/tagVocabulary.json", old)
    assert not ignored("site/backend/app.py", old)
