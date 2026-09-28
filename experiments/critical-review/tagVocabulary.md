# Canonical tag vocabulary (review table)

Generated from `configs/tagVocabulary.json` (built by `build_tag_vocabulary.py`). AUC = frozen-ViT MLP probe on the merged label (OR of members), MyFonts test+val (`validate_tag_vocabulary.py`). Alias weights: 1.0 exact, 0.8 close, 0.5-0.6 loose. Negative weights come only from query negation.

161 canonical tags from 473 MyFonts tags; 185 tags dropped (reasons at the end).

## classification

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| serif | 0.908 | {'serif': 1897, 'roman': 297} | serif, roman | serif (1), serifs (1), serifed (1), with serifs (1), roman (0.8) |  |
| sans-serif | 0.916 | {'sans-serif': 3344, 'sans': 1441, 'sanserif': 409, 'lineal': 52} | sans-serif, sans, sanserif, lineal | sans serif (1), sans-serif (1), sans (1), sanserif (1), sansserif (1), no serifs (1), without serifs (1) |  |
| grotesque | 0.881 | {'grotesk': 610, 'grotesque': 511, 'swiss': 207} | grotesk, grotesque, swiss | grotesque (1), grotesk (1), neo grotesque (1), swiss (0.8), helvetica (0.8) | sans-serif |
| humanist | 0.931 | {'humanist': 582} | humanist | humanist (1) |  |
| geometric | 0.858 | {'geometric': 1833, 'geometric-sans': 62, 'futura': 73} | geometric, geometric-sans, futura | geometric (1), geometric sans (1), futura (0.8), circular (0.5) |  |
| slab-serif | 0.928 | {'slab-serif': 700, 'slab': 293, 'egyptian': 154, 'clarendon': 73} | slab-serif, slab, egyptian, clarendon | slab serif (1), slab (1), slab-serif (1), egyptian (0.8), clarendon (0.8), typewriter serif (0.5) | serif |
| didone | 0.942 | {'didone': 233, 'bodoni': 96} | didone, bodoni | didone (1), didot (1), bodoni (1), modern serif (0.8), fashion serif (0.6) | serif, high-contrast |
| oldstyle | 0.858 | {'oldstyle': 318, 'old-style': 151, 'garalde': 176, 'venetian': 122, 'renaissance': 149} | oldstyle, old-style, garalde, venetian, renaissance | oldstyle (1), old style (1), old-style (1), garalde (1), venetian (1), renaissance (0.8), garamond (0.8) | serif |
| transitional | 0.944 | {'transitional': 229} | transitional | transitional (1), baskerville (0.8) | serif |
| flared | 0.923 | {'flare': 83, 'flare-serif': 52, 'wedge-serif': 73} | flare, flare-serif, wedge-serif | flared (1), flare serif (1), wedge serif (1), glyphic (0.5) |  |
| semi-serif | 0.872 | {'semi-serif': 90, 'some-serifs': 55} | semi-serif, some-serifs | semi serif (1), semi-serif (1) |  |
| inscriptional | 0.823 | {'inscribe': 163, 'glyphic': 75, 'trajan': 68} | inscribe, glyphic, trajan | inscriptional (1), inscribed (1), chiseled (0.8), carved (0.8), trajan (0.8), roman capitals (0.8) | serif |
| blackletter | 0.952 | {'blackletter': 404, 'fraktur': 151, 'textura': 59} | blackletter, fraktur, textura | blackletter (1), black letter (1), fraktur (1), textura (1), old english (1), gothic (0.8), gothic script (0.8), german (0.5) |  |
| uncial | 0.922 | {'uncial': 103, 'irish': 61} | uncial, irish | uncial (1), celtic (0.8), irish (0.8) |  |
| script | 0.934 | {'script': 2309, 'cursive': 940, 'connect': 685} | script, cursive, connect | script (1), cursive (1), joined (0.8), connected (0.8), joined up (0.8), flowing (0.6) |  |
| upright-script | 0.893 | {'upright-script': 145} | upright-script | upright script (1) | script |
| copperplate | 0.940 | {'copperplate': 119, 'spencerian': 60} | copperplate, spencerian | copperplate (1), spencerian (1), engrossing (0.8) | script, calligraphy |
| calligraphy | 0.900 | {'calligraphy': 1163, 'calligraphic': 960, 'quill': 57, 'penmanship': 138} | calligraphy, calligraphic, quill, penmanship | calligraphy (1), calligraphic (1), quill (0.8), penmanship (0.8), nib (0.8), lettering (0.5) |  |
| handwritten | 0.914 | {'handwrite': 2591, 'write': 234, 'pen': 650, 'note': 99, 'notebook': 78} | handwrite, write, pen, note, notebook | handwritten (1), handwriting (1), hand written (1), hand-written (1), handwrite (1), pen (0.8), notes (0.6), notebook (0.6), journal (0.6) |  |
| hand-drawn | 0.856 | {'hand-drawn': 978, 'handmade': 1154, 'hand': 1086, 'handletter': 720, 'handcraft': 112} | hand-drawn, handmade, hand, handletter, handcraft | hand drawn (1), hand-drawn (1), handmade (1), hand made (1), hand lettered (1), hand lettering (1), handlettered (1), hand crafted (0.8), handcrafted (0.8), homemade (0.8) |  |
| child-handwriting | 0.948 | {'child-writing': 59} | child-writing | child handwriting (1), kid handwriting (1), childs handwriting (1), crayon (0.6) | handwritten, kids |
| signature | 0.949 | {'signature': 160} | signature | signature (1), autograph (0.8) | script |
| brush | 0.893 | {'brush': 981, 'brush-drawn': 352, 'brush-script': 132, 'brush-pen': 63, 'dry-brush': 91, 'paint': 265} | brush, brush-drawn, brush-script, brush-pen, dry-brush, paint | brush (1), brush script (1), brushed (1), brush pen (1), dry brush (1), painted (0.8), paint (0.8), watercolor (0.5) |  |
| marker | 0.880 | {'marker': 247, 'felt-tip': 127} | marker, felt-tip | marker (1), felt tip (1), sharpie (0.8), highlighter (0.5) |  |
| chalk | 0.947 | {'chalk': 53} | chalk | chalk (1), chalkboard (1), blackboard (0.8) |  |
| pencil | 0.755 | {'pencil': 74} | pencil | pencil (1), graphite (0.8) |  |
| sketchy | 0.856 | {'sketch': 261, 'scribble': 74, 'doodle': 106, 'draw': 257} | sketch, scribble, doodle, draw | sketch (1), sketchy (1), scribble (1), scribbled (1), doodle (1), doodles (1), drawn (0.6) |  |
| monospace | 0.908 | {'monospace': 151, 'monospaced': 97} | monospace, monospaced | monospace (1), monospaced (1), mono (0.8), fixed width (1), fixed-width (1), coding (0.6), code (0.6), programming (0.6), terminal (0.6) |  |
| typewriter | 0.916 | {'typewriter': 159} | typewriter | typewriter (1), typewritten (1), typed (0.6) | monospace |
| pixel | 0.907 | {'pixel': 96, 'bitmap': 172, 'low-res': 92} | pixel, bitmap, low-res | pixel (1), pixelated (1), pixel art (1), 8 bit (1), 8-bit (1), 8bit (1), bitmap (1), low res (1), retro game (0.6) |  |
| digital | 0.868 | {'lcd': 54, 'digital': 194, 'electronic': 105} | lcd, digital, electronic | digital (1), lcd (1), led (1), digital clock (1), seven segment (1), segment (0.8), electronic (0.8), calculator (0.8) |  |
| dingbat | 0.790 | {'dingbat': 490, 'symbol': 597, 'picture': 439, 'icon': 202, 'non-alphabetic': 212, 'arrow': 154, 'border': 124, 'frame': 124} | dingbat, symbol, picture, icon, non-alphabetic, arrow, border, frame | dingbat (1), dingbats (1), symbol (1), symbols (1), icon (1), icons (1), pictures (0.8), picture font (1), arrows (0.8), borders (0.8), frames (0.8) |  |

## weight

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| bold | 0.770 | {'bold': 1517, 'heavy': 1344, 'black': 551, 'thick': 87} | bold, heavy, black, thick | bold (1), heavy (1), black (0.6), thick (1), heavyweight (1) |  |
| ultra-bold | 0.832 | {'ultra-bold': 117, 'ultra-black': 56, 'fat': 326, 'chunky': 100, 'plump': 73, 'ultra': 81} | ultra-bold, ultra-black, fat, chunky, plump, ultra | ultra bold (1), extra bold (1), extra-bold (1), ultra black (1), fat (1), fat face (1), chunky (1), plump (1), very bold (1), super bold (1) | bold |
| thin | 0.842 | {'thin': 585, 'light': 556, 'hairline': 141} | thin, light, hairline | thin (1), light (0.8), hairline (1), lightweight (1), fine (0.6), skinny strokes (0.8) |  |
| monoline | 0.875 | {'monoline': 707, 'low-contrast': 74, 'linear': 302} | monoline, low-contrast, linear | monoline (1), low contrast (1), even stroke (1), uniform stroke (1), single line (0.8) |  |
| high-contrast | 0.913 | {'high-contrast': 245, 'contrast': 170, 'thick-and-thin': 57} | high-contrast, contrast, thick-and-thin | high contrast (1), thick and thin (1), thick thin (1), contrasty (0.8) |  |

## width

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| condensed | 0.885 | {'condense': 798, 'narrow': 930, 'compress': 95, 'tall': 165, 'skinny': 63} | condense, narrow, compress, tall, skinny | condensed (1), narrow (1), compressed (1), tall (0.8), skinny (0.8), tight (0.5) |  |
| wide | 0.849 | {'wide': 547, 'extend': 139} | wide, extend | wide (1), extended (1), expanded (1), wide letters (1), stretched (0.6) |  |

## case

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| all-caps | 0.723 | {'all-caps': 696, 'caps-only': 311, 'uppercase': 56, 'cap': 345} | all-caps, caps-only, uppercase, cap | all caps (1), all-caps (1), caps only (1), uppercase (1), upper case (1), capitals (0.8), caps (0.8) |  |
| small-caps | 0.791 | {'small-caps': 539} | small-caps | small caps (1), small-caps (1) |  |

## slant

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| italic | 0.809 | {'italic': 471, 'true-italics': 222, 'slant': 131, 'oblique': 89} | italic, true-italics, slant, oblique | italic (1), italics (1), slanted (1), slant (1), oblique (1), leaning (0.8), tilted (0.6) |  |

## proportion

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| large-x-height | 0.803 | {'large-x-height': 66, 'high-x-height': 55} | large-x-height, high-x-height | large x height (1), high x height (1), big x height (1) |  |

## shape

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| rounded | 0.821 | {'round': 1301, 'soft': 606} | round, soft | rounded (1), round (1), rounded corners (1), soft (0.8), bubbly (0.6), bubble (0.6) |  |
| square | 0.864 | {'square': 627, 'squarish': 182, 'rectangular': 112, 'block': 471, 'octagonal': 104, 'polygonal': 78, 'chamfer': 67} | square, squarish, rectangular, block, octagonal, polygonal, chamfer | square (1), squared (1), squarish (1), boxy (1), blocky (1), block (0.8), rectangular (1), octagonal (1), chamfered (1) |  |
| angular | 0.710 | {'angular': 291, 'sharp': 364, 'triangle': 57} | angular, sharp, triangle | angular (1), sharp (1), pointy (1), pointed (0.8), spiky (0.8), jagged (0.8), triangular (0.8) |  |
| modular | 0.822 | {'modular': 290, 'grid': 132, 'construct': 143} | modular, grid, construct | modular (1), grid (0.8), constructed (0.8) |  |
| swash | 0.844 | {'swash': 1054, 'flourish': 307, 'swirl': 103, 'curl': 80, 'curly': 390, 'swash-caps': 57} | swash, flourish, swirl, curl, curly, swash-caps | swash (1), swashes (1), flourish (1), flourishes (1), flourished (1), swirly (1), swirls (1), curly (1), curls (0.8), loopy (0.6) |  |
| irregular | 0.804 | {'irregular': 404, 'uneven': 144, 'messy': 117, 'random': 106, 'jumpy': 51, 'bouncy': 320, 'loose': 80, 'free-form': 98, 'jag': 79} | irregular, uneven, messy, random, jumpy, bouncy, loose, free-form, jag | irregular (1), uneven (1), messy (1), wobbly (0.8), bouncy (1), jumpy (1), loose (0.8), wonky (0.8), imperfect (0.8) |  |
| initials | 0.851 | {'initial': 174, 'monogram': 104} | initial, monogram | initials (1), initial caps (1), drop cap (1), drop caps (1), monogram (1) |  |
| primitive | 0.782 | {'primitive': 112} | primitive | primitive (1), crude (0.8), tribal (0.5) |  |

## texture

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| distressed | 0.849 | {'distress': 372, 'rough': 1069, 'erode': 151, 'damage': 115, 'decay': 69, 'corrode': 65, 'scratch': 70, 'wear': 117, 'noisy': 78, 'texture': 274, 'textured': 182} | distress, rough, erode, damage, decay, corrode, scratch, wear, noisy, texture, textured | distressed (1), rough (1), worn (1), weathered (1), eroded (1), damaged (1), scratched (1), textured (1), texture (0.8), rusty (0.8), aged (0.6) |  |
| grunge | 0.861 | {'grunge': 907, 'dirty': 85, 'splatter': 66, 'punk': 211} | grunge, dirty, splatter, punk | grunge (1), grungy (1), dirty (1), splatter (1), splattered (1), punk (0.8) |  |
| letterpress | 0.746 | {'letterpress': 266, 'stamp': 156} | letterpress, stamp | letterpress (1), stamp (1), stamped (1), rubber stamp (1) |  |
| wood-type | 0.828 | {'wood-type': 428, 'wood': 96, 'tuscan': 138, 'circus': 163} | wood-type, wood, tuscan, circus | wood type (1), woodtype (1), wood (0.6), tuscan (1), circus (0.8), carnival (0.6) |  |

## effect

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| outline | 0.771 | {'outline': 448} | outline | outline (1), outlined (1), hollow (0.8), open (0.5) |  |
| inline | 0.826 | {'inline': 331, 'prismatic': 55} | inline, prismatic | inline (1), prismatic (1), striped (0.6) |  |
| shadow | 0.728 | {'shadow': 248, 'shade': 109} | shadow, shade | shadow (1), shadowed (1), drop shadow (1), shaded (1) |  |
| 3d | 0.774 | {'3d': 277} | 3d | 3d (1), 3-d (1), three dimensional (1), dimensional (0.8) |  |
| layered | 0.750 | {'layer': 241, 'chromatic': 200} | layer, chromatic | layered (1), chromatic (1), multicolor (0.8), color font (0.6) |  |
| stencil | 0.844 | {'stencil': 517} | stencil | stencil (1), stenciled (1), stencilled (1) |  |
| dotted | 0.824 | {'dot': 143} | dot | dotted (1), dots (1), dot matrix (1), polka dot (0.6) |  |
| neon | 0.910 | {'neon': 109} | neon | neon (1), neon sign (1), glowing (0.6) |  |
| distorted | 0.857 | {'distort': 75} | distort | distorted (1), warped (0.8), glitch (0.6), glitchy (0.6) |  |
| cut-out | 0.772 | {'cutup': 64, 'scrap': 68} | cutup, scrap | cut out (1), cutout (1), ransom note (1), collage (0.8), cut up (1) |  |
| engraved | 0.817 | {'engrave': 326} | engrave | engraved (1), engraving (1), etched (0.8) |  |

## decoration

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| ornate | 0.770 | {'ornate': 192, 'ornamental': 355, 'ornament': 1006, 'embellishment': 79, 'decoration': 80} | ornate, ornamental, ornament, embellishment, decoration | ornate (1), ornamental (1), ornamented (1), embellished (1), decorated (0.8), fancy (0.6), decorative (0.6) |  |
| floral | 0.873 | {'flower': 132, 'floral': 85, 'botanical': 95} | flower, floral, botanical | floral (1), flower (1), flowers (1), botanical (1), leaves (0.8), vines (0.8) |  |
| patterned | 0.886 | {'pattern': 197} | pattern | patterned (1), pattern (1) |  |
| ribbon | 0.773 | {'ribbon': 99} | ribbon | ribbon (1), ribbons (1), folded ribbon (1) |  |

## era

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| art-deco | 0.792 | {'art-deco': 890, 'artdeco': 70, 'deco': 276} | art-deco, artdeco, deco | art deco (1), art-deco (1), artdeco (1), deco (1), gatsby (0.8), roaring twenties (0.6) |  |
| art-nouveau | 0.829 | {'art-nouveau': 325} | art-nouveau | art nouveau (1), nouveau (1), jugendstil (1) |  |
| bauhaus | 0.838 | {'bauhaus': 199} | bauhaus | bauhaus (1) |  |
| modernist | 0.788 | {'modernism': 64} | modernism | modernist (1), modernism (1), mid century (0.6), mid-century (0.6) |  |
| constructivist | 0.811 | {'constructivist': 63, 'propaganda': 52} | constructivist, propaganda | constructivist (1), constructivism (1), propaganda (1), soviet (0.8) |  |
| avant-garde | 0.748 | {'avant-garde': 140, 'experimental': 323} | avant-garde, experimental | avant garde (1), avant-garde (1), experimental (1) |  |
| victorian | 0.791 | {'victorian': 189, '1800s': 488, '1890s': 58} | victorian, 1800s, 1890s | victorian (1), 1800s (1), 19th century (1), 1890s (1), steampunk (0.6) |  |
| early-1900s | 0.730 | {'1900s': 309, '1910s': 77} | 1900s, 1910s | 1900s (1), 1910s (1), early 1900s (1), edwardian (0.8) |  |
| 1920s | 0.797 | {'1920s': 451} | 1920s | 1920s (1), 20s (0.8), twenties (0.8) |  |
| 1930s | 0.801 | {'1930s': 585} | 1930s | 1930s (1), 30s (0.8), thirties (0.8) |  |
| 1940s | 0.781 | {'1940s': 397} | 1940s | 1940s (1), 40s (0.8), forties (0.8), wartime (0.6) |  |
| 1950s | 0.752 | {'1950s': 484} | 1950s | 1950s (1), 50s (0.8), fifties (0.8) |  |
| 1960s | 0.719 | {'1960s': 455} | 1960s | 1960s (1), 60s (0.8), sixties (0.8) |  |
| 1970s | 0.764 | {'1970s': 378} | 1970s | 1970s (1), 70s (0.8), seventies (0.8) |  |
| 1980s | 0.654 | {'1980s': 230} | 1980s | 1980s (1), 80s (0.8), eighties (0.8) |  |
| 18th-century | 0.934 | {'1700s': 92} | 1700s | 1700s (1), 18th century (1), colonial (0.6) |  |
| medieval | 0.817 | {'medieval': 105} | medieval | medieval (1), middle ages (1), knight (0.5) |  |
| baroque | 0.903 | {'baroque': 64} | baroque | baroque (1), rococo (0.8) |  |
| historical | 0.764 | {'historical': 127, 'historic': 73, 'ancient': 523, 'classical': 56} | historical, historic, ancient, classical | historical (1), historic (1), ancient (1), classical (0.8) |  |
| vintage | 0.710 | {'vintage': 1601, 'antique': 672, 'old': 250, 'old-fashioned': 80, 'nostalgic': 57, 'quaint': 142} | vintage, antique, old, old-fashioned, nostalgic, quaint | vintage (1), antique (1), old fashioned (1), old-fashioned (1), old (0.6), nostalgic (0.8), classic (0.5) |  |
| retro | 0.716 | {'retro': 2319, 'revival': 195} | retro, revival | retro (1), throwback (0.8), revival (0.6) |  |
| classic | 0.735 | {'classic': 570, 'traditional': 66, 'conservative': 167} | classic, traditional, conservative | classic (1), traditional (1), timeless (0.8), conservative (0.8) |  |
| western | 0.895 | {'western': 260, 'wild-west': 262, 'cowboy': 94, 'rodeo': 63, 'country': 90} | western, wild-west, cowboy, rodeo, country | western (1), wild west (1), cowboy (1), rodeo (1), saloon (0.8), wanted poster (0.8), country (0.6) |  |
| psychedelic | 0.839 | {'psychedelic': 63, 'hippie': 65} | psychedelic, hippie | psychedelic (1), trippy (1), hippie (1), groovy (0.8), flower power (0.8) |  |
| disco | 0.796 | {'disco': 131} | disco | disco (1), funk (0.6) |  |
| hipster | 0.798 | {'hipster': 328} | hipster | hipster (1) |  |
| futuristic | 0.845 | {'futuristic': 589, 'future': 83, 'sci-fi': 143, 'space': 98, 'alien': 53, 'robot': 79} | futuristic, future, sci-fi, space, alien, robot | futuristic (1), future (0.8), sci fi (1), sci-fi (1), scifi (1), rocket (0.6), outer space (1), science fiction (1), space (0.8), alien (0.8), robot (0.8), cyberpunk (0.6) |  |
| techno | 0.893 | {'techno': 534, 'tech': 60, 'technology': 104, 'computer': 440} | techno, tech, technology, computer | techno (1), tech (1), technology (1), computer (1), cyber (0.8) |  |
| technical | 0.805 | {'technical': 565, 'mechanical': 271, 'industrial': 325, 'industry': 83, 'architect': 120, 'architecture': 71, 'din': 115} | technical, mechanical, industrial, industry, architect, architecture, din | technical (1), mechanical (1), industrial (1), engineering (0.8), din (1), architectural (0.8), blueprint (0.6) |  |
| signage | 0.718 | {'signage': 915, 'wayfinding': 64, 'information': 184, 'transport': 96} | signage, wayfinding, information, transport | signage (1), wayfinding (1), road sign (0.8), airport (0.6) |  |
| sign-painting | 0.859 | {'sign-painting': 335, 'showcard': 120} | sign-painting, showcard | sign painting (1), sign painter (1), signwriting (1), showcard (1) |  |
| graffiti | 0.829 | {'graffiti': 191, 'grafitti': 76, 'street': 107} | graffiti, grafitti, street | graffiti (1), grafitti (1), street art (1), tag (0.5), spray paint (0.8) |  |
| urban | 0.736 | {'urban': 302} | urban | urban (1), street (0.6) |  |
| tattoo | 0.774 | {'tattoo': 200} | tattoo | tattoo (1), tattoos (1) |  |
| rock | 0.775 | {'heavy-metal': 67, 'metal': 95, 'rock': 75} | heavy-metal, metal, rock | heavy metal (1), metal (0.8), rock (0.8), rock and roll (0.8), band (0.5) |  |

## theme

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| asian-style | 0.813 | {'asian': 67} | asian | asian (1), chinese (0.8), japanese (0.8), oriental (0.6), chopstick (0.8), asian style (1) |  |
| arabic-style | 0.793 | {'arabic': 93} | arabic | arabic (1), middle eastern (0.8), arabesque (0.6), arabic style (1) |  |
| faux-cyrillic | 0.779 | {'russian': 168} | russian | russian (1), faux cyrillic (1) |  |
| mexican | 0.827 | {'mexican': 56} | mexican | mexican (1), fiesta (0.6) |  |
| african | 0.844 | {'africa': 58} | africa | african (1), africa (1), tribal (0.6) |  |
| horror | 0.827 | {'horror': 179, 'scary': 85, 'evil': 71, 'monster': 98, 'ghost': 131, 'dark': 80} | horror, scary, evil, monster, ghost, dark | horror (1), scary (1), spooky (1), creepy (1), evil (1), monster (1), ghost (1), zombie (0.8), blood (0.8), bloody (0.8), dark (0.6), gothic horror (1) |  |
| halloween | 0.868 | {'halloween': 218} | halloween | halloween (1) | horror |
| fantasy | 0.728 | {'fantasy': 83, 'fairytale': 92, 'magic': 105, 'mysterious': 75} | fantasy, fairytale, magic, mysterious | fantasy (1), fairytale (1), fairy tale (1), magic (1), magical (1), mystical (1), mysterious (0.8), enchanted (0.8), wizard (0.8) |  |
| pirate | 0.877 | {'pirate': 69} | pirate | pirate (1), nautical (0.6) |  |
| christmas | 0.749 | {'christmas': 239, 'xmas': 227, 'winter': 53} | christmas, xmas, winter | christmas (1), xmas (1), holiday (0.6), winter (0.8), snow (0.8), festive (0.6) |  |
| valentine | 0.752 | {'valentine': 415, 'love': 339, 'heart': 108} | valentine, love, heart | valentine (1), valentines (1), love (0.8), hearts (0.8) |  |
| party | 0.733 | {'party': 414, 'birthday': 140, 'celebration': 121, 'festive': 142, 'holiday': 157} | party, birthday, celebration, festive, holiday | party (1), birthday (1), celebration (1), festive (0.8), holiday (0.8) |  |
| wedding | 0.859 | {'wed': 1045} | wed | wedding (1), weddings (1), bridal (1), bride (0.8) |  |
| invitation | 0.800 | {'invitation': 1272, 'invite': 77, 'announcement': 76, 'stationery': 196, 'stationary': 96, 'greeting-card': 139, 'greet': 382, 'card': 186, 'postcard': 122} | invitation, invite, announcement, stationery, stationary, greeting-card, greet, card, postcard | invitation (1), invitations (1), invite (1), stationery (1), greeting card (1), card (0.6), postcard (0.8), announcement (0.8) |  |
| certificate | 0.769 | {'certificate': 192, 'diploma': 88} | certificate, diploma | certificate (1), diploma (1), award (0.6) |  |
| kids | 0.819 | {'kid': 793, 'child': 587, 'childish': 74, 'childrens-book': 64, 'baby': 53, 'toy': 123} | kid, child, childish, childrens-book, baby, toy | kids (1), kid (1), children (1), childrens (1), child (1), childish (1), childlike (1), baby (0.8), toddler (0.8), toy (0.8), storybook (0.8), nursery (0.8) |  |
| school | 0.784 | {'school': 238, 'education': 104} | school, education | school (1), education (1), classroom (1), teacher (0.8) |  |
| comic | 0.849 | {'comic': 910, 'comic-book': 147, 'comic-text': 175, 'cartoon': 480, 'animate': 91, 'animation': 52} | comic, comic-book, comic-text, cartoon, animate, animation | comic (1), comics (1), comic book (1), cartoon (1), cartoony (1), animated (0.8), animation (0.8), manga (0.6), speech bubble (0.8) |  |
| video-game | 0.867 | {'video-game': 127} | video-game | video game (1), videogame (1), gaming (0.8), arcade (0.8), game (0.6) |  |
| sport | 0.742 | {'sport': 350, 'athletic': 53, 'football': 62} | sport, athletic, football | sport (1), sports (1), sporty (1), athletic (1), football (0.8), jersey (0.8) |  |
| varsity | 0.793 | {'college': 70} | college | varsity (1), collegiate (1), college (1) |  |
| baseball-script | 0.886 | {'baseball-script': 52, 'baseball': 65} | baseball-script, baseball | baseball (1), baseball script (1) | script, sport |
| speed | 0.768 | {'fast': 223, 'speed': 62} | fast, speed | speed (1), fast (1), racing (1), motion (0.6) |  |
| military | 0.869 | {'military': 169} | military | military (1), army (1) |  |
| surf | 0.805 | {'surf': 123, 'surfer': 63} | surf, surfer | surf (1), surfing (1), surfer (1) |  |
| tropical | 0.790 | {'tropical': 68, 'beach': 78} | tropical, beach | tropical (1), beach (1), hawaiian (0.8) |  |
| summer | 0.729 | {'summer': 170, 'vacation': 189} | summer, vacation | summer (1), vacation (0.8) |  |
| skateboard | 0.966 | {'skatebord': 53} | skatebord | skateboard (1), skate (1), skater (0.8) |  |
| menu | 0.724 | {'menu': 327, 'restaurant': 180, 'cafe': 66, 'coffee': 99, 'bakery': 55, 'food': 528} | menu, restaurant, cafe, coffee, bakery, food | menu (1), restaurant (1), cafe (1), coffee (1), bakery (1), food (0.8) |  |
| candy | 0.831 | {'candy': 152} | candy | candy (1), sweets (0.8) |  |
| corporate | 0.838 | {'corporate': 833, 'business': 188, 'business-text': 113, 'office': 100, 'identity': 208} | corporate, business, business-text, office, identity | corporate (1), business (1), professional (0.8), office (0.8) |  |
| logo | 0.725 | {'logo': 1857, 'logotype': 648, 'brand': 1356} | logo, logotype, brand | logo (1), logos (1), logotype (1), branding (1), brand (1) |  |
| newspaper | 0.808 | {'news': 628, 'newspaper': 369, 'news-headline': 148, 'editorial': 892, 'newsletter': 164} | news, newspaper, news-headline, editorial, newsletter | newspaper (1), news (1), editorial (1), newsletter (0.8), headline (0.5) |  |
| magazine | 0.783 | {'magazine': 2740} | magazine | magazine (1) |  |
| text | 0.859 | {'text': 1315, 'book-text': 175, 'news-text': 88, 'workhorse': 360, 'small-text': 60, 'functional': 97, 'book': 922} | text, book-text, news-text, workhorse, small-text, functional, book | body text (1), text (0.8), book (0.8), reading (0.8), long text (1), paragraph (0.8), workhorse (1) |  |
| legible | 0.786 | {'legible': 2504, 'readable': 204, 'clear': 234} | legible, readable, clear | legible (1), readable (1), easy to read (1), clear (0.8), readability (1) |  |
| display | 0.683 | {'display': 4233, 'headline': 4129, 'title': 652, 'poster': 3489, 'header': 99, 'billboard': 141, 'advertise': 1211, 'flyer': 162} | display, headline, title, poster, header, billboard, advertise, flyer | display (1), headline (1), headlines (1), title (0.8), titles (0.8), poster (0.8), posters (0.8), heading (0.8), billboard (0.8), advertising (0.6) |  |
| decorative | 0.676 | {'decorative': 5059, 'fancy': 907, 'novelty': 225} | decorative, fancy, novelty | decorative (1), fancy (1), novelty (1) |  |
| fashion | 0.722 | {'fashion': 963, 'fashionable': 810, 'cosmetic': 142, 'beauty': 94, 'perfume': 117, 'glamour': 51, 'jewelry': 69, 'chic': 104} | fashion, fashionable, cosmetic, beauty, perfume, glamour, jewelry, chic | fashion (1), fashionable (1), beauty (1), cosmetics (1), perfume (1), glamour (1), glamorous (1), jewelry (1), chic (1), luxury (0.6), vogue (0.8) |  |

## mood

| canonical | AUC | train fonts | members | aliases | implies |
|---|---|---|---|---|---|
| elegant | 0.808 | {'elegant': 2207, 'graceful': 269, 'beautiful': 291, 'classy': 65, 'sophisticate': 171, 'refine': 58, 'delicate': 358} | elegant, graceful, beautiful, classy, sophisticate, refine, delicate | elegant (1), elegance (1), graceful (1), classy (1), sophisticated (1), refined (1), delicate (0.8), beautiful (0.8), luxury (0.6), luxurious (0.6), upscale (0.6) |  |
| formal | 0.827 | {'formal': 616, 'royal': 76} | formal, royal | formal (1), royal (0.8), regal (0.8), stately (0.8) |  |
| romantic | 0.781 | {'romantic': 429} | romantic | romantic (1), romance (1) |  |
| feminine | 0.777 | {'feminine': 732, 'girly': 196, 'girl': 90, 'woman': 69} | feminine, girly, girl, woman | feminine (1), girly (1), girlish (1), girl (0.8), womanly (1), female (0.8) |  |
| masculine | 0.762 | {'masculine': 421} | masculine | masculine (1), manly (1), male (0.8) |  |
| rugged | 0.739 | {'rugged': 75, 'tough': 76, 'sturdy': 203, 'robust': 71} | rugged, tough, sturdy, robust | rugged (1), tough (1), sturdy (1), robust (1), strong (0.6) |  |
| cute | 0.771 | {'cute': 814, 'sweet': 231, 'lovely': 187, 'pretty': 74, 'charm': 54} | cute, sweet, lovely, pretty, charm | cute (1), sweet (1), adorable (1), lovely (0.8), pretty (0.8), charming (0.8), kawaii (1) |  |
| playful | 0.734 | {'playful': 624, 'fun': 1167, 'whimsical': 136, 'lively': 412, 'happy': 386, 'silly': 60} | playful, fun, whimsical, lively, happy, silly | playful (1), fun (1), whimsical (1), lively (1), happy (1), cheerful (1), silly (1), joyful (1) |  |
| funny | 0.766 | {'funny': 1691, 'crazy': 318, 'wild': 319} | funny, crazy, wild | funny (1), humorous (1), goofy (1), crazy (0.8), wacky (0.8), zany (0.8) |  |
| quirky | 0.693 | {'quirky': 223, 'eccentric': 64, 'unusual': 483, 'weird': 77, 'idiosyncratic': 122, 'funky': 191} | quirky, eccentric, unusual, weird, idiosyncratic, funky | quirky (1), eccentric (1), unusual (1), weird (1), odd (0.8), funky (1), strange (0.8) |  |
| friendly | 0.744 | {'friendly': 1157, 'warm': 110, 'personable': 64} | friendly, warm, personable | friendly (1), warm (1), approachable (1), welcoming (0.8), inviting (0.8) |  |
| casual | 0.797 | {'casual': 1024, 'informal': 1950, 'freestyle': 67} | casual, informal, freestyle | casual (1), informal (1), relaxed (0.8), laid back (0.8), easygoing (0.8) |  |
| modern | 0.728 | {'modern': 2567, 'contemporary': 1527, 'fresh': 362, 'trendy': 126, 'stylish': 578} | modern, contemporary, fresh, trendy, stylish | modern (1), contemporary (1), trendy (1), stylish (0.8), fresh (0.8), current (0.6) |  |
| clean | 0.806 | {'clean': 1519, 'simple': 531, 'plain': 206, 'basic': 73, 'neat': 80, 'crisp': 77, 'minimal': 307, 'neutral': 323, 'modest': 124} | clean, simple, plain, basic, neat, crisp, minimal, neutral, modest | clean (1), simple (1), minimal (1), minimalist (1), minimalistic (1), plain (1), basic (0.8), neat (0.8), crisp (0.8), neutral (0.8), understated (0.8) |  |
| organic | 0.765 | {'organic': 776, 'natural': 347, 'nature': 114} | organic, natural, nature | organic (1), natural (1), nature (0.8), earthy (0.8), eco (0.6) |  |
| rustic | 0.836 | {'rustic': 221} | rustic | rustic (1), farmhouse (0.8), handcrafted (0.5) |  |
| edgy | 0.661 | {'edgy': 103, 'attitude': 53} | edgy, attitude | edgy (1), attitude (0.8), rebellious (0.8), aggressive (0.8) |  |

## Dropped

- **language/character-set support, not visible in latin glyphs**: accent, american, capital-sharp-s, cyrillic, czech, diacritic, dutch, english, european, french, german, greek, italian, latin, latino, multilingual, spanish, turkish, ukrainian, versal-eszett
- **OpenType feature or font-product metadata**: alternate, ball-terminals, collection, contextual-alternates, counterless, extra, family, filled-counters, fraction, free, kern, ligature, linotype-taketype, median-spurs, new, old-style-numerals, oldstyle-figures, opentype, optical-sizes, personal, personalize, pintassilgoprints, pro, regular, spur-serif, spurless, static, stylistic-alternates, superfamily, tail, type, typeface, typography, unicase, want, webfont, wishlist
- **usage context with low visual consistency (AUC < 0.72 or grounding ~0)**: 1990s, 2000s, 2010s, abstract, ad, alternative, angle, animal, app, art, artistic, arts-and-crafts, artsy, authentic, banner, big, blog, book-cover, break, brochure, capital, caption, car, catchword, church, circle, color, commercial, construction, cool, correspondence, cover, craft, creative, curve, curvy, dance, design, disconnect, distinctive, diy, dynamic, easy, economic, emblem, ethnic, exotic, expressive, extreme, fill, film, flair, flash, flow, fluid, gaspipe, graphic, green, head, hybrid, illustration, impact, ink, interlock, jazz, label, letter, letterhead, line, luxury, manual, market, masthead, movie, movie-credits, music, nightclub, nightlife, open, original, package, package-design, poetry, point, press, print, product, product-packaging, publication, publish, quick, quote, sassy, scrapbook, screen, sensible, sexy, sign, smooth, solid, star, strong, style, t-shirt, tag, television, toolkit, travel, tv, unique, upright, useful, valuable, versatile, video, water, weather, web, web-graphics, website, wine, young, youth
- **ambiguous across unrelated styles (blackletter vs. gothic sans vs. horror); 'gothic' aliases to blackletter at 0.8**: gothic
- **merged into a canonical it lowered the AUC of (validate_tag_vocabulary.py); searchable only via aliases**: cut, festive-occasions, game
- **family-specific (AUC 0.96 from one or two families), not a style**: realist
