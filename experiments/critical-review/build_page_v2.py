# Builds the pooled round-2 blind trial page from e24_results.json (run in the work dir that holds tiles/).
import json, base64, re, zlib
R = json.load(open("e24_results.json"))
data = []
for r in R["results"]:
    ids = {k: f"f{zlib.crc32(k.encode()):08x}" for k in r["tiles"]}
    tags = [{"t": m.group(1), "w": float(m.group(2))} for m in (re.match(r"(\S+) ([+-][\d.]+)", t) for t in r["tagsC"])]
    data.append({"q": r["query"], "tags": tags,
                 "sys": {s: [ids[k] for k in r[s]] for s in ("C", "B", "R")},
                 "img": {ids[k]: "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
                         for k, p in r["tiles"].items()}})
import os
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "page_template_v2.html")).read()
page = tpl.replace("__DATA__", json.dumps(data)).replace("__GALLERY__", f"{R['gallery']:,}")
open("font_search_trial_v2.html", "w").write(page); print(len(page) // 1024, "KB")
