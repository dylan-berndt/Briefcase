"""
Downloads zips from the Internet Archive item "1.5_million_font_files_collection" (60.24 GB over 13 zips, ~1.05M
.ttf/.otf files by the zip listings, ~219k unique file names) and extracts only the font files.

    python experiments/internet-archive/fetch.py --list                        # zips, sizes, font counts
    python experiments/internet-archive/fetch.py "Fonts Mega Pack 102.000 Fonts.zip"
    python experiments/internet-archive/fetch.py --all

Layout (all under dataset/, gitignored):
    dataset/internetArchive/zips/<zip name>             the download (resumed with HTTP range requests)
    dataset/internetArchive/fonts/<zip stem>/...        extracted .ttf/.otf/.ttc, original folder structure

Extraction keeps only font files and refuses absolute paths or "..": the archive is untrusted third-party data.
Nested zips/rars inside a zip (a few thousand, mostly in "Worlds Largest Font Collection 600000.zip") and Type 1
fonts (.pfb, ~100k) are not extracted.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile

ITEM = "1.5_million_font_files_collection"
ROOT = os.path.join("dataset", "internetArchive")
FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")


def itemFiles():
    with urllib.request.urlopen(f"https://archive.org/metadata/{ITEM}") as response:
        meta = json.load(response)
    return {f["name"]: f for f in meta["files"] if f["name"].lower().endswith(".zip")}


def download(name, info):
    os.makedirs(os.path.join(ROOT, "zips"), exist_ok=True)
    path = os.path.join(ROOT, "zips", name)
    size = int(info["size"])
    have = os.path.getsize(path) if os.path.exists(path) else 0
    url = f"https://archive.org/download/{ITEM}/{urllib.parse.quote(name)}"
    attempt = 0
    while have < size:
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
            with urllib.request.urlopen(request, timeout=60) as response, open(path, "ab") as f:
                started, startHave = time.time(), have
                while chunk := response.read(1 << 20):
                    f.write(chunk)
                    have += len(chunk)
                    rate = (have - startHave) / max(time.time() - started, 1e-6) / 2 ** 20
                    print(f"\r  {name}: {have / 2 ** 30:.2f}/{size / 2 ** 30:.2f} GiB  {rate:.1f} MiB/s", end="", flush=True)
            print()
        except OSError as e:
            attempt += 1
            if attempt > 6:
                raise
            print(f"\n  retry {attempt} after {e}", flush=True)
            time.sleep(2 ** attempt)
            have = os.path.getsize(path)
    if have != size:
        raise RuntimeError(f"{path}: {have} bytes, expected {size}")
    if "md5" in info:
        digest = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 24):
                digest.update(chunk)
        if digest.hexdigest() != info["md5"]:
            raise RuntimeError(f"{path}: md5 mismatch (delete it and re-run)")
    return path


def safeMember(member):
    name = member.filename.replace("\\", "/")
    parts = name.split("/")
    if name.startswith("/") or ".." in parts or (len(parts[0]) == 2 and parts[0][1] == ":"):
        return None
    return name if name.lower().endswith(FONT_EXTENSIONS) and not member.is_dir() else None


def extract(zipPath):
    stem = os.path.splitext(os.path.basename(zipPath))[0]
    out = os.path.join(ROOT, "fonts", stem)
    done = os.path.join(out, ".extracted")
    if os.path.exists(done):
        print(f"  {out} already extracted")
        return out
    count = skipped = 0
    with zipfile.ZipFile(zipPath) as z:
        for member in z.infolist():
            name = safeMember(member)
            if name is None:
                skipped += 1
                continue
            target = os.path.join(out, *name.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            try:
                with z.open(member) as src, open(target, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                count += 1
            except (zipfile.BadZipFile, OSError, NotImplementedError) as e:
                print(f"\n  skipped {name}: {e}")
            if count % 2000 == 0:
                print(f"\r  extracted {count} font files", end="", flush=True)
    print(f"\r  extracted {count} font files ({skipped} other members skipped) -> {out}")
    open(done, "w").close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zips", nargs="*", help="zip names as listed by --list")
    p.add_argument("--all", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--keepZip", action="store_true", help="keep the zip after extracting (default: delete it)")
    args = p.parse_args()

    files = itemFiles()
    if args.list:
        for name, info in sorted(files.items(), key=lambda kv: -int(kv[1]["size"])):
            print(f"{int(info['size']) / 2 ** 30:7.2f} GiB  {name}")
        return
    names = sorted(files) if args.all else args.zips
    for name in names:
        if name not in files:
            sys.exit(f"unknown zip {name!r}; see --list")
        print(name, flush=True)
        if os.path.exists(os.path.join(ROOT, "fonts", os.path.splitext(name)[0], ".extracted")):
            print("  already extracted, skipping download")
            continue
        zipPath = download(name, files[name])
        extract(zipPath)
        if not args.keepZip:
            os.remove(zipPath)


if __name__ == "__main__":
    main()
