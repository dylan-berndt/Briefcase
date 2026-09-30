"""
Downloads Flickr8k (8,091 images, 5 human captions each -- ~40k pairs, the
same order of magnitude as this project's own ~40k-font / ~220k-query
corpus) from the community GitHub mirror (the original UIUC/Illinois host
is down; this mirror is the standard replacement used by most current
tutorials). Used as a KNOWN-WORKING reference point for the question:
why does short-text-query image retrieval work well in general, but not
for fonts? See README.md in this folder for the full experimental design.

    python3 experiments/embedding-geometry/download_flickr8k.py
"""
import os
import zipfile

import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "flickr8k")
FILES = {
    "Flickr8k_Dataset.zip": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
    "Flickr8k_text.zip": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
}


def download(url, path):
    if os.path.exists(path):
        print(f"already have {path}")
        return
    print(f"downloading {url} ...")
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r{written / 1e6:.1f}/{total / 1e6:.1f} MB", end="")
    print()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in FILES.items():
        zipPath = os.path.join(DATA_DIR, name)
        download(url, zipPath)
        print(f"extracting {name} ...")
        with zipfile.ZipFile(zipPath) as z:
            z.extractall(DATA_DIR)

    images = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".jpeg"))]
    print(f"\n{len(images)} images extracted to {DATA_DIR}")
    tokenFile = os.path.join(DATA_DIR, "Flickr8k.token.txt")
    if os.path.exists(tokenFile):
        print(f"caption file present: {tokenFile}")
    else:
        # some mirrors nest text files one directory deeper
        for root, _, files in os.walk(DATA_DIR):
            if "Flickr8k.token.txt" in files:
                print(f"caption file present: {os.path.join(root, 'Flickr8k.token.txt')}")
                break
        else:
            print("WARNING: Flickr8k.token.txt not found, check the extracted contents")


if __name__ == "__main__":
    main()
