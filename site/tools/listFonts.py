"""Step 1 of building the search bundle: decide which fonts go on the site.

    python site/tools/listFonts.py --google google/fonts --dafont dafont --dafontPageList dafont/cache/font_list.json \\
        --out build/fonts.json
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fontList import listFonts  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--google", default=os.path.join("google", "fonts"))
    parser.add_argument("--dafont", default="dafont")
    parser.add_argument("--dafontPageList", default=os.path.join("dafont", "cache", "font_list.json"),
                        help="dafonts-free cache/font_list.json: each family's DaFont page and creator")
    parser.add_argument("--out", default=os.path.join("build", "fonts.json"))
    args = parser.parse_args()

    pageList = args.dafontPageList if os.path.exists(args.dafontPageList) else None
    if args.dafont and pageList is None:
        print(f"warning: {args.dafontPageList} not found, DaFont fonts have no page to link to and are all skipped")

    fonts, skipped = listFonts(args.google if os.path.isdir(args.google) else None,
                               args.dafont if os.path.isdir(args.dafont) else None, pageList)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as file:
        json.dump(fonts, file, indent=1)

    print(f"{len(fonts)} fonts: {dict(Counter(f['source'] for f in fonts))}; left out: {skipped}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
