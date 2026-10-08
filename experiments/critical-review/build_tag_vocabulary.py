# Draft canonical tag vocabulary + alias table for tag search (future-work items 11-12 in CLAUDE.md).
#
# Source vocabulary: MyFonts tags with >=20 training fonts (see e22_vocab.py). All >=50-font tags and the
# 20-49-font tags with AUC >= 0.75 are decided by hand below; the rest of the 20-49 band is left out. Each canonical
# tag lists the raw MyFonts tags merged into it ("members"; the training label is the OR of members),
# the phrases a user might type ("aliases", weight 1.0 = means exactly this tag, 0.8 = close synonym,
# 0.5-0.6 = loose/related), broader tags it implies, and a facet for grouping in the UI.
# Every decided candidate is either a member of exactly one canonical tag or listed in DROPPED
# with a reason; the script asserts this.
#
#   python experiments/critical-review/build_tag_vocabulary.py --candidates candidates.tsv
#
# candidates.tsv: tag \t trainFonts \t tagger ROC-AUC \t descriptorGrounding score (from e22_vocab.py).
# Writes configs/tagVocabulary.json -- the MERGED vocabulary (186 canonicals). The committed configs/tagVocabulary.json has since
# been regrouped to spelling variants only by site/tools/regroupVocabulary.py, so running this overwrites it; point --out elsewhere.

import argparse
import json
import os

# canonical: (facet, members, aliases {phrase: weight}, implies)
CANONICAL = {
    # ---- classification ----
    "serif": ("classification", ["serif", "roman"],
              {"serif": 1.0, "serifs": 1.0, "serifed": 1.0, "with serifs": 1.0, "roman": 0.8}, []),
    "sans-serif": ("classification", ["sans-serif", "sans", "sanserif", "lineal"],
                   {"sans serif": 1.0, "sans-serif": 1.0, "sans": 1.0, "sanserif": 1.0, "sansserif": 1.0,
                    "no serifs": 1.0, "without serifs": 1.0}, []),
    "grotesque": ("classification", ["grotesk", "grotesque", "swiss"],
                  {"grotesque": 1.0, "grotesk": 1.0, "neo grotesque": 1.0, "swiss": 0.8, "helvetica": 0.8}, ["sans-serif"]),
    "humanist": ("classification", ["humanist"], {"humanist": 1.0}, []),
    "geometric": ("classification", ["geometric", "geometric-sans", "futura"],
                  {"geometric": 1.0, "geometric sans": 1.0, "futura": 0.8, "circular": 0.5}, []),
    "slab-serif": ("classification", ["slab-serif", "slab", "egyptian", "clarendon"],
                   {"slab serif": 1.0, "slab": 1.0, "slab-serif": 1.0, "egyptian": 0.8, "clarendon": 0.8,
                    "typewriter serif": 0.5}, ["serif"]),
    "didone": ("classification", ["didone", "bodoni"],
               {"didone": 1.0, "didot": 1.0, "bodoni": 1.0, "modern serif": 0.8, "fashion serif": 0.6}, ["serif", "high-contrast"]),
    "oldstyle": ("classification", ["oldstyle", "old-style", "garalde", "venetian", "renaissance"],
                 {"oldstyle": 1.0, "old style": 1.0, "old-style": 1.0, "garalde": 1.0, "venetian": 1.0,
                  "renaissance": 0.8, "garamond": 0.8}, ["serif"]),
    "transitional": ("classification", ["transitional"], {"transitional": 1.0, "baskerville": 0.8}, ["serif"]),
    "flared": ("classification", ["flare", "flare-serif", "wedge-serif"],
               {"flared": 1.0, "flare serif": 1.0, "wedge serif": 1.0, "glyphic": 0.5}, []),
    "semi-serif": ("classification", ["semi-serif", "some-serifs"], {"semi serif": 1.0, "semi-serif": 1.0}, []),
    "inscriptional": ("classification", ["inscribe", "glyphic", "trajan"],
                      {"inscriptional": 1.0, "inscribed": 1.0, "chiseled": 0.8, "carved": 0.8, "trajan": 0.8,
                       "roman capitals": 0.8}, ["serif"]),
    "blackletter": ("classification", ["blackletter", "fraktur", "textura"],
                    {"blackletter": 1.0, "black letter": 1.0, "fraktur": 1.0, "textura": 1.0, "old english": 1.0,
                     "gothic": 0.8, "gothic script": 0.8, "german": 0.5}, []),
    "uncial": ("classification", ["uncial", "irish"], {"uncial": 1.0, "celtic": 0.6, "irish": 0.8}, []),
    "script": ("classification", ["script", "cursive", "connect"],
               {"script": 1.0, "cursive": 1.0, "joined": 0.8, "connected": 0.8, "joined up": 0.8,
                "flowing": 0.6}, []),
    "upright-script": ("classification", ["upright-script"], {"upright script": 1.0}, ["script"]),
    "copperplate": ("classification", ["copperplate", "spencerian"],
                    {"copperplate": 1.0, "spencerian": 1.0, "engrossing": 0.8}, ["script", "calligraphy"]),
    "calligraphy": ("classification", ["calligraphy", "calligraphic", "quill", "penmanship"],
                    {"calligraphy": 1.0, "calligraphic": 1.0, "quill": 0.8, "penmanship": 0.8,
                     "nib": 0.8, "lettering": 0.5}, []),
    "handwritten": ("classification", ["handwrite", "write", "pen", "note", "notebook"],
                    {"handwritten": 1.0, "handwriting": 1.0, "hand written": 1.0, "hand-written": 1.0,
                     "handwrite": 1.0, "pen": 0.8, "notes": 0.6, "notebook": 0.6, "journal": 0.6}, []),
    "hand-drawn": ("classification", ["hand-drawn", "handmade", "hand", "handletter", "handcraft"],
                   {"hand drawn": 1.0, "hand-drawn": 1.0, "handmade": 1.0, "hand made": 1.0,
                    "hand lettered": 1.0, "hand lettering": 1.0, "handlettered": 1.0, "hand crafted": 0.8,
                    "handcrafted": 0.8, "homemade": 0.8}, []),
    "child-handwriting": ("classification", ["child-writing"],
                          {"child handwriting": 1.0, "kid handwriting": 1.0, "childs handwriting": 1.0,
                           "crayon": 0.6}, ["handwritten", "kids"]),
    "signature": ("classification", ["signature"], {"signature": 1.0, "autograph": 0.8}, ["script"]),
    "brush": ("classification", ["brush", "brush-drawn", "brush-script", "brush-pen", "dry-brush", "paint"],
              {"brush": 1.0, "brush script": 1.0, "brushed": 1.0, "brush pen": 1.0, "dry brush": 1.0,
               "painted": 0.8, "paint": 0.8, "watercolor": 0.5}, []),
    "marker": ("classification", ["marker", "felt-tip"],
               {"marker": 1.0, "felt tip": 1.0, "sharpie": 0.8, "highlighter": 0.5}, []),
    "chalk": ("classification", ["chalk"], {"chalk": 1.0, "chalkboard": 1.0, "blackboard": 0.8}, []),
    "pencil": ("classification", ["pencil"], {"pencil": 1.0, "graphite": 0.8}, []),
    "sketchy": ("classification", ["sketch", "scribble", "doodle", "draw"],
                {"sketch": 1.0, "sketchy": 1.0, "scribble": 1.0, "scribbled": 1.0, "doodle": 1.0,
                 "doodles": 1.0, "drawn": 0.6}, []),
    "monospace": ("classification", ["monospace", "monospaced"],
                  {"monospace": 1.0, "monospaced": 1.0, "mono": 0.8, "fixed width": 1.0, "fixed-width": 1.0,
                   "coding": 0.6, "code": 0.6, "programming": 0.6, "terminal": 0.6}, []),
    "typewriter": ("classification", ["typewriter"], {"typewriter": 1.0, "typewritten": 1.0, "typed": 0.6},
                   ["monospace"]),
    "pixel": ("classification", ["pixel", "bitmap", "low-res"],
              {"pixel": 1.0, "pixelated": 1.0, "pixel art": 1.0, "8 bit": 1.0, "8-bit": 1.0, "8bit": 1.0,
               "bitmap": 1.0, "low res": 1.0, "retro game": 0.6}, []),
    "digital": ("classification", ["lcd", "digital", "electronic"],
                {"digital": 1.0, "lcd": 1.0, "led": 1.0, "digital clock": 1.0, "seven segment": 1.0,
                 "segment": 0.8, "electronic": 0.8, "calculator": 0.8}, []),
    "dingbat": ("classification", ["dingbat", "symbol", "picture", "icon", "non-alphabetic", "arrow", "border", "frame"],
                {"dingbat": 1.0, "dingbats": 1.0, "symbol": 1.0, "symbols": 1.0, "icon": 1.0, "icons": 1.0,
                 "pictures": 0.8, "picture font": 1.0, "arrows": 0.8, "borders": 0.8, "frames": 0.8}, []),

    # ---- weight / contrast ----
    "bold": ("weight", ["bold", "heavy", "black", "thick"],
             {"bold": 1.0, "heavy": 1.0, "black": 0.6, "thick": 1.0, "heavyweight": 1.0}, []),
    "ultra-bold": ("weight", ["ultra-bold", "ultra-black", "fat", "chunky", "plump", "ultra"],
                   {"ultra bold": 1.0, "extra bold": 1.0, "extra-bold": 1.0, "ultra black": 1.0, "fat": 1.0,
                    "chunky": 1.0, "plump": 1.0, "very bold": 1.0, "super bold": 1.0},
                   ["bold"]),
    "thin": ("weight", ["thin", "light", "hairline"],
             {"thin": 1.0, "light": 0.8, "hairline": 1.0, "lightweight": 1.0, "fine": 0.6, "skinny strokes": 0.8}, []),
    "monoline": ("weight", ["monoline", "low-contrast", "linear"],
                 {"monoline": 1.0, "low contrast": 1.0, "even stroke": 1.0, "uniform stroke": 1.0,
                  "single line": 0.8}, []),
    "high-contrast": ("weight", ["high-contrast", "contrast", "thick-and-thin"],
                      {"high contrast": 1.0, "thick and thin": 1.0, "thick thin": 1.0, "contrasty": 0.8}, []),

    # ---- width / proportion / case / slant ----
    "condensed": ("width", ["condense", "narrow", "compress", "tall", "skinny"],
                  {"condensed": 1.0, "narrow": 1.0, "compressed": 1.0, "tall": 0.8, "skinny": 0.8,
                   "tight": 0.5}, []),
    "wide": ("width", ["wide", "extend"],
             {"wide": 1.0, "extended": 1.0, "expanded": 1.0, "wide letters": 1.0, "stretched": 0.6}, []),
    "all-caps": ("case", ["all-caps", "caps-only", "uppercase", "cap"],
                 {"all caps": 1.0, "all-caps": 1.0, "caps only": 1.0, "uppercase": 1.0, "upper case": 1.0,
                  "capitals": 0.8, "caps": 0.8}, []),
    "small-caps": ("case", ["small-caps"], {"small caps": 1.0, "small-caps": 1.0}, []),
    "italic": ("slant", ["italic", "true-italics", "slant", "oblique"],
               {"italic": 1.0, "italics": 1.0, "slanted": 1.0, "slant": 1.0, "oblique": 1.0, "leaning": 0.8,
                "tilted": 0.6}, []),
    "large-x-height": ("proportion", ["large-x-height", "high-x-height"],
                       {"large x height": 1.0, "high x height": 1.0, "big x height": 1.0}, []),

    # ---- shape ----
    "rounded": ("shape", ["round", "soft"],
                {"rounded": 1.0, "round": 1.0, "rounded corners": 1.0, "soft": 0.8,
                 "bubble": 0.6}, []),
    "square": ("shape", ["square", "squarish", "rectangular", "block", "octagonal", "polygonal", "chamfer"],
               {"square": 1.0, "squared": 1.0, "squarish": 1.0, "boxy": 1.0, "blocky": 1.0, "block": 0.8,
                "rectangular": 1.0, "octagonal": 1.0, "chamfered": 1.0}, []),
    "angular": ("shape", ["angular", "sharp", "triangle"],
                {"angular": 1.0, "sharp": 1.0, "pointy": 1.0, "pointed": 0.8, "spiky": 0.8, "jagged": 0.8,
                 "triangular": 0.8}, []),
    "modular": ("shape", ["modular", "grid", "construct"],
                {"modular": 1.0, "grid": 0.8, "constructed": 0.8}, []),
    "swash": ("shape", ["swash", "flourish", "swirl", "curl", "curly", "swash-caps"],
              {"swash": 1.0, "swashes": 1.0, "flourish": 1.0, "flourishes": 1.0, "flourished": 1.0,
               "swirly": 1.0, "swirls": 1.0, "curly": 1.0, "curls": 0.8, "loopy": 0.6}, []),
    "irregular": ("shape", ["irregular", "uneven", "messy", "random", "jumpy", "bouncy", "loose", "free-form", "jag"],
                  {"irregular": 1.0, "uneven": 1.0, "messy": 1.0, "wobbly": 0.8, "bouncy": 1.0, "jumpy": 1.0,
                   "loose": 0.8, "wonky": 0.8, "imperfect": 0.8}, []),
    "initials": ("shape", ["initial", "monogram"],
                 {"initials": 1.0, "initial caps": 1.0, "drop cap": 1.0, "drop caps": 1.0, "monogram": 1.0}, []),

    # ---- texture / effect ----
    "distressed": ("texture", ["distress", "rough", "erode", "damage", "decay", "corrode", "scratch", "wear",
                               "noisy", "texture", "textured"],
                   {"distressed": 1.0, "rough": 1.0, "worn": 1.0, "weathered": 1.0, "eroded": 1.0,
                    "damaged": 1.0, "scratched": 1.0, "textured": 1.0, "texture": 0.8, "rusty": 0.8,
                    "aged": 0.6}, []),
    "grunge": ("texture", ["grunge", "dirty", "splatter", "punk"],
               {"grunge": 1.0, "grungy": 1.0, "dirty": 1.0, "splatter": 1.0, "splattered": 1.0, "punk": 0.8}, []),
    "letterpress": ("texture", ["letterpress", "stamp"],
                    {"letterpress": 1.0, "stamp": 1.0, "stamped": 1.0, "rubber stamp": 1.0}, []),
    "wood-type": ("texture", ["wood-type", "wood", "tuscan", "circus"],
                  {"wood type": 1.0, "woodtype": 1.0, "wood": 0.6, "tuscan": 1.0, "circus": 0.8,
                   "carnival": 0.6}, []),
    "outline": ("effect", ["outline"], {"outline": 1.0, "outlined": 1.0, "hollow": 0.8, "open": 0.5}, []),
    "inline": ("effect", ["inline", "prismatic"], {"inline": 1.0, "prismatic": 1.0, "striped": 0.6}, []),
    "shadow": ("effect", ["shadow", "shade"], {"shadow": 1.0, "shadowed": 1.0, "drop shadow": 1.0,
                                               "shaded": 1.0}, []),
    "3d": ("effect", ["3d"], {"3d": 1.0, "3-d": 1.0, "three dimensional": 1.0, "dimensional": 0.8}, []),
    "layered": ("effect", ["layer", "chromatic"], {"layered": 1.0, "chromatic": 1.0, "multicolor": 0.8,
                                                   "color font": 0.6}, []),
    "stencil": ("effect", ["stencil"], {"stencil": 1.0, "stenciled": 1.0, "stencilled": 1.0}, []),
    "dotted": ("effect", ["dot"], {"dotted": 1.0, "dots": 1.0, "polka dot": 0.6}, []),
    "neon": ("effect", ["neon"], {"neon": 1.0, "neon sign": 1.0, "glowing": 0.6}, []),
    "distorted": ("effect", ["distort"], {"distorted": 1.0, "warped": 0.8, "glitch": 0.6, "glitchy": 0.6}, []),
    "cut-out": ("effect", ["cutup", "scrap"],
                {"cut out": 1.0, "cutout": 1.0, "ransom note": 1.0, "collage": 0.8, "cut up": 1.0}, []),
    "engraved": ("effect", ["engrave"], {"engraved": 1.0, "engraving": 1.0, "etched": 0.8}, []),
    "ornate": ("decoration", ["ornate", "ornamental", "ornament", "embellishment", "decoration"],
               {"ornate": 1.0, "ornamental": 1.0, "ornamented": 1.0, "embellished": 1.0, "decorated": 0.8,
                "fancy": 0.6, "decorative": 0.6}, []),
    "floral": ("decoration", ["flower", "floral", "botanical"],
               {"floral": 1.0, "flower": 1.0, "flowers": 1.0, "botanical": 1.0, "leaves": 0.8, "vines": 0.8}, []),
    "patterned": ("decoration", ["pattern"], {"patterned": 1.0, "pattern": 1.0}, []),
    "primitive": ("shape", ["primitive"], {"primitive": 1.0, "crude": 0.8, "tribal": 0.5}, []),
    "ribbon": ("decoration", ["ribbon"], {"ribbon": 1.0, "ribbons": 1.0, "folded ribbon": 1.0}, []),

    # ---- era / style ----
    "art-deco": ("era", ["art-deco", "artdeco", "deco"],
                 {"art deco": 1.0, "art-deco": 1.0, "artdeco": 1.0, "deco": 1.0, "gatsby": 0.8,
                  "roaring twenties": 0.6}, []),
    "art-nouveau": ("era", ["art-nouveau"], {"art nouveau": 1.0, "nouveau": 1.0, "jugendstil": 1.0}, []),
    "bauhaus": ("era", ["bauhaus"], {"bauhaus": 1.0}, []),
    "modernist": ("era", ["modernism"], {"modernist": 1.0, "modernism": 1.0, "mid century": 0.6,
                                         "mid-century": 0.6}, []),
    "constructivist": ("era", ["constructivist", "propaganda"],
                       {"constructivist": 1.0, "constructivism": 1.0, "propaganda": 1.0, "soviet": 0.8}, []),
    "avant-garde": ("era", ["avant-garde", "experimental"],
                    {"avant garde": 1.0, "avant-garde": 1.0, "experimental": 1.0}, []),
    "victorian": ("era", ["victorian", "1800s", "1890s"],
                  {"victorian": 1.0, "1800s": 1.0, "19th century": 1.0, "1890s": 1.0, "steampunk": 0.6}, []),
    "early-1900s": ("era", ["1900s", "1910s"], {"1900s": 1.0, "1910s": 1.0, "early 1900s": 1.0,
                                               "edwardian": 0.8}, []),
    "1920s": ("era", ["1920s"], {"1920s": 1.0, "20s": 0.8, "twenties": 0.8}, []),
    "1930s": ("era", ["1930s"], {"1930s": 1.0, "30s": 0.8, "thirties": 0.8}, []),
    "1940s": ("era", ["1940s"], {"1940s": 1.0, "40s": 0.8, "forties": 0.8, "wartime": 0.6}, []),
    "1950s": ("era", ["1950s"], {"1950s": 1.0, "50s": 0.8, "fifties": 0.8}, []),
    "1960s": ("era", ["1960s"], {"1960s": 1.0, "60s": 0.8, "sixties": 0.8}, []),
    "1970s": ("era", ["1970s"], {"1970s": 1.0, "70s": 0.8, "seventies": 0.8}, []),
    "1980s": ("era", ["1980s"], {"1980s": 1.0, "80s": 0.8, "eighties": 0.8}, []),
    "18th-century": ("era", ["1700s"], {"1700s": 1.0, "18th century": 1.0, "colonial": 0.6}, []),
    "medieval": ("era", ["medieval"], {"medieval": 1.0, "middle ages": 1.0, "knight": 0.5}, []),
    "baroque": ("era", ["baroque"], {"baroque": 1.0, "rococo": 0.8}, []),
    "historical": ("era", ["historical", "historic", "ancient", "classical"],
                   {"historical": 1.0, "historic": 1.0, "ancient": 1.0, "classical": 0.8}, []),
    "vintage": ("era", ["vintage", "antique", "old", "old-fashioned", "nostalgic", "quaint"],
                {"vintage": 1.0, "antique": 1.0, "old fashioned": 1.0, "old-fashioned": 1.0, "old": 0.6,
                 "nostalgic": 0.8, "classic": 0.5}, []),
    "retro": ("era", ["retro", "revival"], {"retro": 1.0, "throwback": 0.8, "revival": 0.6}, []),
    "classic": ("era", ["classic", "traditional", "conservative"],
                {"classic": 1.0, "traditional": 1.0, "timeless": 0.8, "conservative": 0.8}, []),
    "western": ("era", ["western", "wild-west", "cowboy", "rodeo", "country"],
                {"western": 1.0, "wild west": 1.0, "cowboy": 1.0, "rodeo": 1.0, "saloon": 0.8,
                 "wanted poster": 0.8, "country": 0.6}, []),
    "psychedelic": ("era", ["psychedelic", "hippie"], {"psychedelic": 1.0, "trippy": 1.0, "hippie": 1.0,
                                                       "groovy": 0.8, "flower power": 0.8}, []),
    "disco": ("era", ["disco"], {"disco": 1.0, "funk": 0.6}, []),
    "hipster": ("era", ["hipster"], {"hipster": 1.0}, []),
    "futuristic": ("era", ["futuristic", "future", "sci-fi", "space", "alien", "robot"],
                   {"futuristic": 1.0, "future": 0.8, "sci fi": 1.0, "sci-fi": 1.0, "scifi": 1.0, "rocket": 0.6, "outer space": 1.0,
                    "science fiction": 1.0, "space": 0.8, "alien": 0.8, "robot": 0.8, "cyberpunk": 0.6}, []),
    "techno": ("era", ["techno", "tech", "technology", "computer"],
               {"techno": 1.0, "tech": 1.0, "technology": 1.0, "computer": 1.0, "cyber": 0.8}, []),
    "technical": ("era", ["technical", "mechanical", "industrial", "industry", "architect", "architecture", "din"],
                  {"technical": 1.0, "mechanical": 1.0, "industrial": 1.0, "engineering": 0.8, "din": 1.0,
                   "architectural": 0.8, "blueprint": 0.6}, []),
    "signage": ("era", ["signage", "wayfinding", "information", "transport"],
                {"signage": 1.0, "wayfinding": 1.0, "road sign": 0.8, "airport": 0.6}, []),
    "sign-painting": ("era", ["sign-painting", "showcard"],
                      {"sign painting": 1.0, "sign painter": 1.0, "signwriting": 1.0, "showcard": 1.0}, []),
    "graffiti": ("era", ["graffiti", "grafitti", "street"],
                 {"graffiti": 1.0, "grafitti": 1.0, "street art": 1.0, "tag": 0.5, "spray paint": 0.6}, []),
    "urban": ("era", ["urban"], {"urban": 1.0, "street": 0.6}, []),
    "tattoo": ("era", ["tattoo"], {"tattoo": 1.0, "tattoos": 1.0}, []),
    "rock": ("era", ["heavy-metal", "metal", "rock"], {"heavy metal": 1.0, "metal": 0.8, "rock": 0.8,
                                                        "rock and roll": 0.8, "band": 0.5}, []),

    # ---- themes / cultures ----
    "asian-style": ("theme", ["asian"], {"asian": 1.0, "chinese": 0.8, "japanese": 0.8, "oriental": 0.6,
                                         "chopstick": 0.8}, []),
    "arabic-style": ("theme", ["arabic"], {"arabic": 1.0, "middle eastern": 0.8, "arabesque": 0.6}, []),
    "faux-cyrillic": ("theme", ["russian"], {"russian": 1.0, "faux cyrillic": 1.0}, []),
    "mexican": ("theme", ["mexican"], {"mexican": 1.0, "fiesta": 0.6}, []),
    "african": ("theme", ["africa"], {"african": 1.0, "africa": 1.0, "tribal": 0.6}, []),
    "horror": ("theme", ["horror", "scary", "evil", "monster", "ghost", "dark"],
               {"horror": 1.0, "scary": 1.0, "spooky": 1.0, "creepy": 1.0, "evil": 1.0, "monster": 1.0,
                "ghost": 1.0, "zombie": 0.8, "blood": 0.8, "bloody": 0.8, "dark": 0.6, "gothic horror": 1.0}, []),
    "halloween": ("theme", ["halloween"], {"halloween": 1.0}, ["horror"]),
    "fantasy": ("theme", ["fantasy", "magic", "mysterious"],
                {"fantasy": 1.0, "magic": 1.0, "mystical": 1.0, "mysterious": 0.8, "wizard": 0.8}, []),
    "pirate": ("theme", ["pirate"], {"pirate": 1.0, "nautical": 0.6}, []),
    "christmas": ("theme", ["christmas", "xmas", "winter"],
                  {"christmas": 1.0, "xmas": 1.0, "holiday": 0.6, "winter": 0.8, "snow": 0.8, "festive": 0.6}, []),
    "valentine": ("theme", ["valentine", "love", "heart"],
                  {"valentine": 1.0, "valentines": 1.0, "love": 0.8, "hearts": 0.8}, []),
    "party": ("theme", ["party", "birthday", "celebration", "festive", "holiday"],
              {"party": 1.0, "birthday": 1.0, "celebration": 1.0, "festive": 0.8, "holiday": 0.8}, []),
    "wedding": ("theme", ["wed"], {"wedding": 1.0, "weddings": 1.0, "bridal": 1.0, "bride": 0.8}, []),
    "invitation": ("theme", ["invitation", "invite", "announcement", "stationery", "stationary",
                             "greeting-card", "greet", "card", "postcard"],
                   {"invitation": 1.0, "invitations": 1.0, "invite": 1.0, "stationery": 1.0,
                    "greeting card": 1.0, "card": 0.6, "postcard": 0.8, "announcement": 0.8}, []),
    "certificate": ("theme", ["certificate", "diploma"], {"certificate": 1.0, "diploma": 1.0,
                                                          "award": 0.6}, []),
    "kids": ("theme", ["kid", "child", "childish", "childrens-book", "baby", "toy"],
             {"kids": 1.0, "kid": 1.0, "children": 1.0, "childrens": 1.0, "child": 1.0, "childish": 1.0,
              "childlike": 1.0, "baby": 0.8, "toddler": 0.8, "toy": 0.8, "storybook": 0.8,
              "nursery": 0.8}, []),
    "school": ("theme", ["school", "education"], {"school": 1.0, "education": 1.0, "classroom": 1.0,
                                                  "teacher": 0.8}, []),
    "comic": ("theme", ["comic", "comic-book", "comic-text", "cartoon", "animate", "animation"],
              {"comic": 1.0, "comics": 1.0, "comic book": 1.0, "cartoon": 1.0, "cartoony": 1.0,
               "animated": 0.8, "animation": 0.8, "manga": 0.6, "speech bubble": 0.8}, []),
    "video-game": ("theme", ["video-game"], {"video game": 1.0, "videogame": 1.0, "gaming": 0.8,
                                             "arcade": 0.8, "game": 0.6}, []),
    "sport": ("theme", ["sport", "athletic", "football"],
              {"sport": 1.0, "sports": 1.0, "sporty": 1.0, "athletic": 1.0, "football": 0.8, "jersey": 0.8}, []),
    "varsity": ("theme", ["college"], {"varsity": 1.0, "collegiate": 1.0, "college": 1.0}, []),
    "baseball-script": ("theme", ["baseball-script", "baseball"], {"baseball": 1.0}, ["script", "sport"]),
    "speed": ("theme", ["fast", "speed"], {"speed": 1.0, "fast": 1.0, "racing": 1.0, "motion": 0.6}, []),
    "military": ("theme", ["military"], {"military": 1.0, "army": 1.0}, []),
    "surf": ("theme", ["surf", "surfer"], {"surf": 1.0, "surfing": 1.0, "surfer": 1.0}, []),
    "tropical": ("theme", ["tropical", "beach"], {"tropical": 1.0, "beach": 1.0, "hawaiian": 0.8}, []),
    "summer": ("theme", ["summer", "vacation"], {"summer": 1.0, "vacation": 0.8}, []),
    "skateboard": ("theme", ["skatebord"], {"skateboard": 1.0, "skate": 1.0, "skater": 0.8}, []),
    "menu": ("theme", ["menu", "restaurant", "cafe", "coffee", "bakery", "food"],
             {"menu": 1.0, "restaurant": 1.0, "cafe": 1.0, "coffee": 1.0, "bakery": 1.0, "food": 0.8}, []),
    "candy": ("theme", ["candy"], {"candy": 1.0, "sweets": 0.8}, []),
    "corporate": ("theme", ["corporate", "business", "business-text", "office", "identity"],
                  {"corporate": 1.0, "business": 1.0, "professional": 0.8, "office": 0.8}, []),
    "logo": ("theme", ["logo", "logotype", "brand"], {"logo": 1.0, "logos": 1.0, "logotype": 1.0,
                                                    "branding": 1.0, "brand": 1.0}, []),
    "newspaper": ("theme", ["news", "newspaper", "news-headline", "editorial", "newsletter"],
                  {"newspaper": 1.0, "news": 1.0, "editorial": 1.0, "newsletter": 0.8, "headline": 0.5}, []),
    "magazine": ("theme", ["magazine"], {"magazine": 1.0}, []),
    "text": ("theme", ["text", "book-text", "news-text", "workhorse", "small-text", "functional", "book"],
             {"body text": 1.0, "text": 0.8, "book": 0.8, "reading": 0.8, "long text": 1.0,
              "paragraph": 0.8, "workhorse": 1.0}, []),
    "legible": ("theme", ["legible", "readable", "clear"],
                {"legible": 1.0, "readable": 1.0, "easy to read": 1.0, "clear": 0.8, "readability": 1.0}, []),
    "display": ("theme", ["display", "headline", "title", "poster", "header", "billboard", "advertise", "flyer"],
                {"display": 1.0, "headline": 1.0, "headlines": 1.0, "title": 0.8, "titles": 0.8,
                 "poster": 0.8, "posters": 0.8, "heading": 0.8, "billboard": 0.8, "advertising": 0.6}, []),
    "decorative": ("theme", ["decorative", "fancy", "novelty"],
                   {"decorative": 1.0, "fancy": 1.0, "novelty": 1.0}, []),
    "fashion": ("theme", ["fashion", "fashionable", "cosmetic", "beauty", "perfume", "glamour", "jewelry", "chic"],
                {"fashion": 1.0, "fashionable": 1.0, "beauty": 1.0, "cosmetics": 1.0, "perfume": 1.0,
                 "glamour": 1.0, "glamorous": 1.0, "jewelry": 1.0, "chic": 1.0, "luxury": 0.6,
                 "vogue": 0.8}, []),

    # ---- mood ----
    "elegant": ("mood", ["elegant", "graceful", "beautiful", "classy", "sophisticate", "refine", "delicate"],
                {"elegant": 1.0, "elegance": 1.0, "graceful": 1.0, "classy": 1.0, "sophisticated": 1.0,
                 "refined": 1.0, "delicate": 0.8, "beautiful": 0.8, "luxury": 0.6, "luxurious": 0.6,
                 "upscale": 0.6}, []),
    "formal": ("mood", ["formal", "royal"], {"formal": 1.0, "royal": 0.8, "regal": 0.8, "stately": 0.8}, []),
    "romantic": ("mood", ["romantic"], {"romantic": 1.0, "romance": 1.0}, []),
    "feminine": ("mood", ["feminine", "girly", "girl", "woman"],
                 {"feminine": 1.0, "girly": 1.0, "girlish": 1.0, "girl": 0.8, "womanly": 1.0, "female": 0.8}, []),
    "masculine": ("mood", ["masculine"], {"masculine": 1.0, "manly": 1.0, "male": 0.8}, []),
    "rugged": ("mood", ["rugged", "tough", "sturdy", "robust"],
               {"rugged": 1.0, "tough": 1.0, "sturdy": 1.0, "robust": 1.0, "strong": 0.6}, []),
    "cute": ("mood", ["cute", "sweet", "lovely", "pretty", "charm"],
             {"cute": 1.0, "sweet": 1.0, "adorable": 1.0, "lovely": 0.8, "pretty": 0.8, "charming": 0.8,
              "kawaii": 1.0}, []),
    "playful": ("mood", ["playful", "fun", "whimsical", "lively", "happy", "silly"],
                {"playful": 1.0, "fun": 1.0, "whimsical": 1.0, "lively": 1.0, "happy": 1.0, "cheerful": 1.0,
                 "silly": 1.0, "joyful": 1.0}, []),
    "funny": ("mood", ["funny", "crazy", "wild"], {"funny": 1.0, "humorous": 1.0, "goofy": 1.0,
                                                   "crazy": 0.8, "wacky": 0.8, "zany": 0.8}, []),
    "quirky": ("mood", ["quirky", "eccentric", "unusual", "weird", "idiosyncratic", "funky"],
               {"quirky": 1.0, "eccentric": 1.0, "unusual": 1.0, "weird": 1.0, "odd": 0.8, "funky": 1.0,
                "strange": 0.8}, []),
    "friendly": ("mood", ["friendly", "warm", "personable"],
                 {"friendly": 1.0, "warm": 1.0, "approachable": 1.0, "welcoming": 0.8, "inviting": 0.8}, []),
    "casual": ("mood", ["casual", "informal", "freestyle"],
               {"casual": 1.0, "informal": 1.0, "relaxed": 0.8, "laid back": 0.8, "easygoing": 0.8}, []),
    "modern": ("mood", ["modern", "contemporary", "fresh", "trendy", "stylish"],
               {"modern": 1.0, "contemporary": 1.0, "trendy": 1.0, "stylish": 0.8, "fresh": 0.8,
                "current": 0.6}, []),
    "clean": ("mood", ["clean", "simple", "plain", "basic", "neat", "crisp", "minimal", "neutral", "modest"],
              {"clean": 1.0, "simple": 1.0, "minimal": 1.0, "minimalist": 1.0, "minimalistic": 1.0,
               "plain": 1.0, "basic": 0.8, "neat": 0.8, "crisp": 0.8, "neutral": 0.8, "understated": 0.8}, []),
    "organic": ("mood", ["organic", "natural", "nature"], {"organic": 1.0, "natural": 1.0, "nature": 0.8,
                                                          "earthy": 0.8, "eco": 0.6}, []),
    "rustic": ("mood", ["rustic"], {"rustic": 1.0, "farmhouse": 0.8,
                                                 "handcrafted": 0.5}, []),
    "edgy": ("mood", ["edgy", "attitude"], {"edgy": 1.0, "attitude": 0.8, "rebellious": 0.8, "aggressive": 0.8}, []),
}

# Every other >=50-font candidate, with the reason it's not searchable.
DROPPED = {}
def drop(reason, *tags):
    for t in tags:
        DROPPED[t] = reason

drop("language/character-set support, not visible in latin glyphs",
     "multilingual", "cyrillic", "german", "capital-sharp-s", "versal-eszett", "latin", "greek", "english", "french",
     "italian", "spanish", "dutch", "czech", "turkish", "ukrainian", "european", "american", "diacritic", "latino")
drop("OpenType feature or font-product metadata",
     "opentype", "ligature", "alternate", "contextual-alternates", "stylistic-alternates", "kern", "fraction",
     "old-style-numerals", "oldstyle-figures", "optical-sizes", "family", "superfamily", "collection", "static",
     "regular", "free", "pro", "webfont", "unicase", "tail", "ball-terminals", "counterless", "filled-counters",
     "spurless", "median-spurs", "spur-serif", "linotype-taketype", "pintassilgoprints", "type", "typeface",
     "typography", "new", "extra", "personal", "personalize", "wishlist", "want")
drop("usage context with low visual consistency (AUC < 0.72 or grounding ~0)",
     "commercial", "label", "package", "package-design", "product", "product-packaging", "t-shirt",
     "cover", "book-cover", "letterhead", "banner", "ad", "movie", "film", "tv", "television", "video",
     "movie-credits", "music", "jazz", "dance", "nightclub", "nightlife", "travel", "craft", "arts-and-crafts",
     "diy", "blog", "website", "app", "web", "web-graphics", "screen", "catchword", "quote", "tag", "caption",
     "masthead", "publish", "publication", "brochure", "correspondence", "letter", "print", "press", "sign",
     "impact", "car", "market", "wine", "water", "weather", "green", "animal", "star", "circle", "curve",
     "curvy", "point", "head", "big", "color", "open", "solid", "smooth", "flow", "fluid",
     "line", "capital", "upright", "illustration", "graphic", "art", "design", "creative", "artistic",
     "artsy", "style", "unique", "original", "distinctive", "alternative", "hybrid", "abstract", "exotic",
     "ethnic", "cool", "strong", "dynamic", "angle", "extreme", "easy", "useful",
     "versatile", "sensible", "economic", "toolkit", "manual", "authentic", "youth", "young", "2000s", "1990s",
     "2010s", "sexy", "flair", "sassy", "expressive", "valuable",
     "emblem", "quick", "fill", "break", "disconnect", "gaspipe", "ink", "scrapbook", "interlock",
     "flash", "church", "construction")
drop("ambiguous across unrelated styles (blackletter vs. gothic sans vs. horror); 'gothic' aliases to blackletter at 0.8",
     "gothic")
drop("usage context with low visual consistency (AUC < 0.72 or grounding ~0)", "luxury", "poetry")
drop("language/character-set support, not visible in latin glyphs", "accent")
drop("merged into a canonical it lowered the AUC of (validate_tag_vocabulary.py); searchable only via aliases",
     "game", "cut", "festive-occasions")
drop("family-specific (AUC 0.96 from one or two families), not a style", "realist")



# ---- additions from the 20-49-font band (reviewed after the round-2 trial's "cute bubbly" miss) ----
# Band tags are noisier (AUC measured on 5-25 held-out positives), so they mostly join an existing
# canonical as extra members; new canonicals only where the style has no home yet.
def extend(canon, members=(), aliases=None):
    facet, m, a, implies = CANONICAL[canon]
    CANONICAL[canon] = (facet, m + list(members), {**a, **(aliases or {})}, implies)

extend("serif", ["antiqua", "roman-serif"], {"antiqua": 0.8})
extend("grotesque", ["neo-grotesque"])
extend("sans-serif", ["linear-sans"])
extend("humanist", ["humanistic"], {"humanistic": 1.0})
extend("geometric", ["circular"])
extend("slab-serif", ["egyptienne"], {"egyptienne": 0.8})
extend("didone", ["didot", "neoclassical"], {"neoclassical": 0.8})
extend("transitional", ["baskerville", "scotch"], {"scotch roman": 1.0, "scotch": 0.6})
extend("oldstyle", ["garamond"])
extend("flared", ["flared-serif", "stressed-sans"], {"stressed sans": 1.0, "optima": 0.8})
extend("semi-serif", ["tiny-serif"], {"tiny serifs": 1.0, "micro serifs": 1.0})
extend("inscriptional", ["lapidary"], {"lapidary": 1.0})
extend("script", ["join", "all-connecting", "casual-script"], {"casual script": 1.0, "all connecting": 1.0})
extend("calligraphy", ["nib", "fountain"], {"nib": 0.8, "pen nib": 0.8, "fountain pen": 0.8, "dip pen": 0.8})
extend("hand-drawn", ["freehand"], {"freehand": 1.0, "free hand": 1.0})
extend("brush", ["brush-font", "brush-lettering", "brushstroke"], {"brushstroke": 1.0, "brush stroke": 1.0,
                                                                   "brush lettering": 1.0})
extend("chalk", ["chalkboard"])
extend("sketchy", ["scratchy"], {"scratchy": 0.8})
extend("monospace", ["code"], {"coding": 0.8, "code": 0.8, "programming": 1.0, "programmer": 1.0})
extend("dingbat", ["pictogram", "clip-art", "silhouette", "fleuron", "fleurons"],
       {"pictogram": 1.0, "pictograms": 1.0, "clip art": 1.0, "clipart": 1.0, "silhouettes": 0.8,
        "fleuron": 1.0, "fleurons": 1.0, "printer ornaments": 1.0})
extend("ultra-bold", ["extra-bold"])
extend("monoline", ["monolinear"], {"monolinear": 1.0})
extend("condensed", ["ultra-narrow"], {"ultra narrow": 1.0, "extra condensed": 1.0, "compact": 0.6})
extend("wide", ["expand"])
extend("italic", ["true-italic"])
extend("square", ["blocky", "boxy", "eurostile"], {"blocky": 1.0, "boxy": 1.0, "eurostile": 0.8})
extend("swash", ["swirly"], {"swirly": 1.0})
extend("irregular", ["wobbly"], {"wobbly": 1.0, "wonky": 0.8, "shaky": 0.8})
extend("grunge", ["destroy", "destruct", "trash"], {"destroyed": 1.0, "trashed": 0.8, "wrecked": 0.8})
extend("cut-out", ["cut-out"])
extend("letterpress", ["rubber-stamp"], {"rubber stamp": 1.0})
extend("neon", ["tube"], {"neon tube": 1.0})
extend("ribbon", ["streamer"], {"streamer": 0.8})
extend("art-nouveau", ["jugendstil"], {"jugendstil": 1.0})
extend("constructivist", ["soviet"], {"soviet": 1.0, "ussr": 0.8})
extend("victorian", ["1880s"], {"1880s": 1.0})
extend("early-1900s", ["edwardian"])
extend("medieval", ["monastic", "manuscript", "incunabula"],
       {"monastic": 0.8, "illuminated manuscript": 1.0, "manuscript": 0.8, "incunabula": 0.8})
extend("psychedelic", ["groovy"])
extend("futuristic", ["scifi", "robotic"], {"robotic": 1.0})
extend("techno", ["hi-tech", "high-tech"], {"hi tech": 1.0, "high tech": 1.0})
extend("technical", ["industrial-sans", "machinery", "mechanic", "railroad"],
       {"machinery": 0.8, "machine": 0.8, "railroad": 0.6, "railway": 0.6})
extend("signage", ["highway", "traffic", "metro"],
       {"highway": 1.0, "road sign": 1.0, "traffic sign": 1.0, "metro": 0.8, "subway": 0.8, "transit": 0.8})
extend("urban", ["hip-hop"], {"hip hop": 1.0, "rap": 0.8})
extend("tattoo", ["tatoo"])
extend("asian-style", ["far-east"], {"far east": 1.0, "bamboo": 0.8})
extend("faux-cyrillic", ["old-russian"], {"old russian": 1.0})
extend("horror", ["spooky", "creepy", "vampire", "death", "skull", "skeleton", "blood"],
       {"vampire": 1.0, "skull": 0.8, "skeleton": 0.8, "death": 0.8})
extend("halloween", ["witch"], {"witch": 0.8, "witches": 0.8})
extend("fantasy", ["wizard"], {"wizard": 0.8})
extend("christmas", ["snow", "snowflake"], {"snowflake": 0.8, "snowy": 0.8})
extend("sport", ["basketball", "hockey"], {"basketball": 1.0, "hockey": 1.0})
extend("tropical", ["tiki"], {"tiki": 1.0, "hawaiian": 0.8})
extend("corporate", ["corporative", "professional"])
extend("magazine", ["magazin"])
extend("fashion", ["perfums"])
extend("cute", ["adorable"], {"adorable": 1.0})
extend("funny", ["comedy"], {"comedy": 1.0, "comedic": 1.0})
extend("quirky", ["offbeat"], {"offbeat": 1.0})
extend("clean", ["minimalist"])
extend("wedding", [], {"save the date": 1.0, "anniversary": 0.6})
extend("menu", [], {"recipe": 0.6, "diner": 0.6, "gourmet": 0.6})
extend("romantic", [], {"sensual": 0.6})

CANONICAL.update({
    "bubble": ("shape", ["bubble", "bubbly", "bulbous"],
               {"bubble": 1.0, "bubbles": 1.0, "bubbly": 1.0, "bulbous": 1.0, "puffy": 0.8, "balloon": 0.8,
                "inflated": 0.8}, []),
    "fat-face": ("classification", ["fat-face"], {"fat face": 1.0, "fatface": 1.0}, ["ultra-bold", "high-contrast"]),
    "reverse-contrast": ("weight", ["reverse-contrast"],
                         {"reverse contrast": 1.0, "reversed contrast": 1.0, "reverse stress": 1.0}, []),
    "chancery": ("classification", ["chancery"], {"chancery": 1.0, "chancery italic": 1.0, "italic hand": 0.8},
                 ["calligraphy"]),
    "modern-calligraphy": ("classification", ["modern-calligraphy"],
                           {"modern calligraphy": 1.0, "brush calligraphy": 0.8}, ["calligraphy"]),
    "ink-traps": ("shape", ["ink-traps"], {"ink traps": 1.0, "ink trap": 1.0, "inktraps": 1.0}, []),
    "lowercase": ("case", ["lowercase"], {"lowercase": 1.0, "lower case": 1.0, "all lowercase": 1.0,
                                          "no capitals": 1.0}, []),
    "dot-matrix": ("classification", ["dot-matrix"], {"dot matrix": 1.0, "dotmatrix": 1.0, "led": 0.6,
                                                      "receipt": 0.6}, []),
    "ocr": ("classification", ["ocr"], {"ocr": 1.0, "machine readable": 1.0, "ocr a": 1.0, "ocr b": 1.0}, []),
    "user-interface": ("theme", ["ui", "user-interface", "interface", "apps"],
                       {"ui": 1.0, "user interface": 1.0, "interface": 1.0, "app": 0.8, "apps": 0.8,
                        "dashboard": 0.6}, []),
    "stitched": ("texture", ["stitch", "needlework"],
                 {"stitch": 1.0, "stitched": 1.0, "stitching": 1.0, "embroidery": 1.0, "embroidered": 1.0,
                  "cross stitch": 1.0, "needlework": 1.0, "sewing": 0.8, "sewn": 0.8}, []),
    "watercolor": ("texture", ["watercolor"], {"watercolor": 1.0, "watercolour": 1.0}, []),
    "crayon": ("texture", ["crayon"], {"crayon": 1.0, "crayons": 1.0, "wax crayon": 1.0}, []),
    "spray-paint": ("texture", ["spray-paint", "spray"],
                    {"spray paint": 1.0, "spray painted": 1.0, "spraypaint": 1.0, "aerosol": 1.0, "spray": 0.8}, []),
    "spatter": ("texture", ["spatter"], {"spatter": 1.0, "spattered": 1.0, "ink splatter": 0.8}, []),
    "chrome": ("effect", ["chrome"], {"chrome": 1.0, "metallic": 0.8, "shiny": 0.6}, []),
    "marquee": ("effect", ["marquee"], {"marquee": 1.0, "light bulbs": 1.0, "marquee lights": 1.0,
                                        "broadway": 0.6}, []),
    "striped": ("effect", ["multi-line", "stripe", "strip"],
                {"striped": 1.0, "stripes": 1.0, "stripe": 1.0, "multi line": 1.0, "multiline": 1.0,
                 "parallel lines": 1.0}, []),
    "streamline": ("era", ["streamline"], {"streamline": 1.0, "streamlined": 1.0, "streamline moderne": 1.0},
                   ["art-deco"]),
    "early-modern": ("era", ["1500s", "1600s"],
                     {"1500s": 1.0, "1600s": 1.0, "16th century": 1.0, "17th century": 1.0}, []),
    "celtic": ("theme", ["celtic"], {"celtic": 1.0, "irish": 0.6, "gaelic": 0.8, "insular": 0.8}, []),
    "runic": ("theme", ["rune"], {"rune": 1.0, "runes": 1.0, "runic": 1.0, "viking": 0.6, "norse": 0.6}, []),
    "occult": ("theme", ["occult"], {"occult": 1.0, "witchcraft": 0.8, "tarot": 0.8, "esoteric": 0.8,
                                     "mystic": 0.6}, []),
    "fairy-tale": ("theme", ["fairytale", "fairy", "fairy-tale", "tale", "magical"],
                   {"fairy tale": 1.0, "fairytale": 1.0, "fairy": 1.0, "fairies": 1.0, "enchanted": 0.8,
                    "magical": 0.8}, []),
    "folk": ("theme", ["folk"], {"folk": 1.0, "folk art": 1.0, "folksy": 0.8}, []),
})

drop("20-49 band: language, place or character-set support",
     "%d0%ba%d0%b8%d1%80%d0%b8%d0%bb%d0%bb%d0%b8%d1%86%d0%b0", "cyr", "bulgarian", "central-europe", "polish",
     "romanian", "slovak", "urdu", "persian", "vietnamese", "eszett", "british", "italy", "paris", "america",
     "cuba", "chile", "chilean", "latin-american", "japan", "chinese")
drop("20-49 band: foundry, family or product metadata (or >=50% one foundry prefix)",
     "bluemlein", "bluemlein-script-collection", "scrapper", "teacher", "dandy", "software", "latinotype",
     "daniel-hernandez", "viergutz", "rsz", "pap", "pack", "bundle", "companion", "fav", "favorite", "feature",
     "opentype-features", "old-style-figures", "tabular", "number", "glyph", "element", "letterbat", "motif",
     "personal-text", "generic", "standard", "normal", "medium", "universal", "tech-pubs", "seduce", "creamy",
     "no-baseline", "no-counter", "bi-form", "grayletter", "lead", "century")
drop("20-49 band: usage context or subject, not a visual style",
     "action", "adventure", "alcohol", "beer", "cocktail", "tea", "chocolate", "meal", "recipe", "gourmet",
     "diner", "anniversary", "save-the-date", "dinner-invitation", "badge", "bible", "booklet", "catalogue",
     "cartography", "halftone", "bamboo", "gift", "shop", "retail", "money", "math", "science", "infographic", "fax", "printer",
     "memo", "diary", "people", "bird", "plant", "tree", "sun", "cross", "spring", "zodiac", "vignette",
     "gangster", "fiction", "history", "period", "teen", "teenage", "teenager", "scrapbooking", "body",
     "character", "funeral", "tool", "economical", "compact", "sixty", "textile", "vector", "drop", "reverse", "hands-on")
drop("20-49 band: mood or quality word too vague to search as a tag",
     "agile", "alive", "bounce", "loud", "eye-catcher", "sensationalist", "personality", "spontaneous",
     "relax", "serious", "fine", "fanciful", "movement", "raw", "coarse", "rocky", "rigid", "regal", "sensual",
     "boho", "shabby-chic", "human", "large-aperture", "informal-text", "mono", "slim", "loop", "spiral",
     "feather", "wire", "hand-cut", "hand-painted", "inky", "upright-italic", "expressionist", "vernacular")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", default=os.path.join("configs", "tagVocabulary.json"))
    args = p.parse_args()

    stats = {}
    for line in open(args.candidates):
        tag, n, auc, g = line.rstrip("\n").split("\t")
        stats[tag] = {"trainFonts": int(n), "auc": float(auc), "grounding": float(g)}

    memberOf = {}
    for canon, (facet, members, aliases, implies) in CANONICAL.items():
        for m in members:
            assert m in stats, f"{canon}: member {m!r} is not a >=20-font MyFonts tag"
            assert m not in memberOf, f"{m!r} is in both {memberOf[m]} and {canon}"
            memberOf[m] = canon
        for parent in implies:
            assert parent in CANONICAL, f"{canon} implies unknown {parent!r}"

    # Tags with >=50 fonts, and 20-49-font tags whose AUC is >=0.75, must each be decided explicitly;
    # the rest of the 20-49 band is left out automatically.
    for t, st in stats.items():
        if st["trainFonts"] < 50 and st["auc"] < 0.75 and t not in memberOf:
            DROPPED.setdefault(t, "20-49 band: tagger AUC < 0.75")
    dropped = {t: r for t, r in DROPPED.items() if t in stats and t not in memberOf}
    missing = sorted(set(stats) - set(memberOf) - set(dropped))
    assert not missing, f"undecided candidate tags: {missing}"

    vocab = {}
    for canon, (facet, members, aliases, implies) in CANONICAL.items():
        aliases = {a.lower(): w for a, w in aliases.items()}
        aliases.setdefault(canon.replace("-", " "), 1.0)
        best = max(members, key=lambda m: stats[m]["auc"])
        vocab[canon] = {
            "facet": facet,
            "members": members,
            "aliases": aliases,
            "implies": implies,
            "trainFonts": {m: stats[m]["trainFonts"] for m in members},
            "memberAuc": {m: stats[m]["auc"] for m in members},
            "bestMemberAuc": stats[best]["auc"],
            "grounding": {m: stats[m]["grounding"] for m in members},
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"canonical": vocab, "dropped": dropped}, f, indent=1)
    print(f"{len(vocab)} canonical tags from {len(memberOf)} MyFonts tags; {len(dropped)} dropped -> {args.out}")


if __name__ == "__main__":
    main()
