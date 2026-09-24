import json, base64, html
R=json.load(open("e20_results.json"))
tiles={}
def uri(p):
    if p not in tiles: tiles[p]="data:image/png;base64,"+base64.b64encode(open(p,"rb").read()).decode()
    return tiles[p]
data=[]
for r in R["results"]:
    data.append({"q":r["query"],"tags":r["matchedTags"],
                 "sys":{s:[uri(x["tile"]) for x in r[s]] for s in ("A","B","R")}})
page=open("page_template.html").read().replace("__DATA__",json.dumps(data)).replace("__GALLERY__",str(R["gallery"]))
out="./font_search_trial.html"
open(out,"w").write(page); print(out, len(page)//1024,"KB")
