# Canonical tag vocabulary (review table)

Generated from `configs/tagVocabulary.json` (built by `build_tag_vocabulary.py`). AUC = frozen-ViT MLP probe on the merged label (OR of members), MyFonts test+val (`validate_tag_vocabulary.py`). Alias weights: 1.0 exact, 0.8 close, 0.5-0.6 loose. Negative weights come only from query negation.

186 canonical tags from 606 MyFonts tags (all tags with >=50 training fonts reviewed, plus the 20-49-font tags with AUC >= 0.75); 465 tags dropped (reasons at the end).

## classification

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| serif | 0.911 | serif (1897), roman (297), antiqua (31), roman-serif (42) | serif (1), serifs (1), serifed (1), with serifs (1), roman (0.8), antiqua (0.8) |  |
| sans-serif | 0.916 | sans-serif (3344), sans (1441), sanserif (409), lineal (52), linear-sans (46) | sans serif (1), sans-serif (1), sans (1), sanserif (1), sansserif (1), no serifs (1), without serifs (1) |  |
| grotesque | 0.874 | grotesk (610), grotesque (511), swiss (207), neo-grotesque (37) | grotesque (1), grotesk (1), neo grotesque (1), swiss (0.8), helvetica (0.8) | sans-serif |
| humanist | 0.921 | humanist (582), humanistic (47) | humanist (1), humanistic (1) |  |
| geometric | 0.857 | geometric (1833), geometric-sans (62), futura (73), circular (49) | geometric (1), geometric sans (1), futura (0.8), circular (0.5) |  |
| slab-serif | 0.928 | slab-serif (700), slab (293), egyptian (154), clarendon (73), egyptienne (34) | slab serif (1), slab (1), slab-serif (1), egyptian (0.8), clarendon (0.8), typewriter serif (0.5), egyptienne (0.8) | serif |
| didone | 0.940 | didone (233), bodoni (96), didot (47), neoclassical (27) | didone (1), didot (1), bodoni (1), modern serif (0.8), fashion serif (0.6), neoclassical (0.8) | serif, high-contrast |
| oldstyle | 0.863 | oldstyle (318), old-style (151), garalde (176), venetian (122), renaissance (149), garamond (44) | oldstyle (1), old style (1), old-style (1), garalde (1), venetian (1), renaissance (0.8), garamond (0.8) | serif |
| transitional | 0.952 | transitional (229), baskerville (22), scotch (25) | transitional (1), baskerville (0.8), scotch roman (1), scotch (0.6) | serif |
| flared | 0.933 | flare (83), flare-serif (52), wedge-serif (73), flared-serif (20), stressed-sans (45) | flared (1), flare serif (1), wedge serif (1), glyphic (0.5), stressed sans (1), optima (0.8) |  |
| semi-serif | 0.884 | semi-serif (90), some-serifs (55), tiny-serif (46) | semi serif (1), semi-serif (1), tiny serifs (1), micro serifs (1) |  |
| inscriptional | 0.817 | inscribe (163), glyphic (75), trajan (68), lapidary (27) | inscriptional (1), inscribed (1), chiseled (0.8), carved (0.8), trajan (0.8), roman capitals (0.8), lapidary (1) | serif |
| blackletter | 0.957 | blackletter (404), fraktur (151), textura (59) | blackletter (1), black letter (1), fraktur (1), textura (1), old english (1), gothic (0.8), gothic script (0.8), german (0.5) |  |
| uncial | 0.922 | uncial (103), irish (61) | uncial (1), celtic (0.6), irish (0.8) |  |
| script | 0.934 | script (2309), cursive (940), connect (685), join (43), all-connecting (22), casual-script (22) | script (1), cursive (1), joined (0.8), connected (0.8), joined up (0.8), flowing (0.6), casual script (1), all connecting (1) |  |
| upright-script | 0.890 | upright-script (145) | upright script (1) | script |
| copperplate | 0.954 | copperplate (119), spencerian (60) | copperplate (1), spencerian (1), engrossing (0.8) | script, calligraphy |
| calligraphy | 0.906 | calligraphy (1163), calligraphic (960), quill (57), penmanship (138), nib (43), fountain (26) | calligraphy (1), calligraphic (1), quill (0.8), penmanship (0.8), nib (0.8), lettering (0.5), pen nib (0.8), fountain pen (0.8), dip pen (0.8) |  |
| handwritten | 0.914 | handwrite (2591), write (234), pen (650), note (99), notebook (78) | handwritten (1), handwriting (1), hand written (1), hand-written (1), handwrite (1), pen (0.8), notes (0.6), notebook (0.6), journal (0.6) |  |
| hand-drawn | 0.856 | hand-drawn (978), handmade (1154), hand (1086), handletter (720), handcraft (112), freehand (32) | hand drawn (1), hand-drawn (1), handmade (1), hand made (1), hand lettered (1), hand lettering (1), handlettered (1), hand crafted (0.8), handcrafted (0.8), homemade (0.8), freehand (1), free hand (1) |  |
| child-handwriting | 0.947 | child-writing (59) | child handwriting (1), kid handwriting (1), childs handwriting (1), crayon (0.6) | handwritten, kids |
| signature | 0.944 | signature (160) | signature (1), autograph (0.8) | script |
| brush | 0.891 | brush (981), brush-drawn (352), brush-script (132), brush-pen (63), dry-brush (91), paint (265), brush-font (24), brush-lettering (23), brushstroke (44) | brush (1), brush script (1), brushed (1), brush pen (1), dry brush (1), painted (0.8), paint (0.8), watercolor (0.5), brushstroke (1), brush stroke (1), brush lettering (1) |  |
| marker | 0.880 | marker (247), felt-tip (127) | marker (1), felt tip (1), sharpie (0.8), highlighter (0.5) |  |
| chalk | 0.942 | chalk (53), chalkboard (44) | chalk (1), chalkboard (1), blackboard (0.8) |  |
| pencil | 0.772 | pencil (74) | pencil (1), graphite (0.8) |  |
| sketchy | 0.851 | sketch (261), scribble (74), doodle (106), draw (257), scratchy (32) | sketch (1), sketchy (1), scribble (1), scribbled (1), doodle (1), doodles (1), drawn (0.6), scratchy (0.8) |  |
| monospace | 0.886 | monospace (151), monospaced (97), code (29) | monospace (1), monospaced (1), mono (0.8), fixed width (1), fixed-width (1), coding (0.8), code (0.8), programming (1), terminal (0.6), programmer (1) |  |
| typewriter | 0.922 | typewriter (159) | typewriter (1), typewritten (1), typed (0.6) | monospace |
| pixel | 0.902 | pixel (96), bitmap (172), low-res (92) | pixel (1), pixelated (1), pixel art (1), 8 bit (1), 8-bit (1), 8bit (1), bitmap (1), low res (1), retro game (0.6) |  |
| digital | 0.855 | lcd (54), digital (194), electronic (105) | digital (1), lcd (1), led (1), digital clock (1), seven segment (1), segment (0.8), electronic (0.8), calculator (0.8) |  |
| dingbat | 0.796 | dingbat (490), symbol (597), picture (439), icon (202), non-alphabetic (212), arrow (154), border (124), frame (124), pictogram (36), clip-art (31), silhouette (48), fleuron (41), fleurons (34) | dingbat (1), dingbats (1), symbol (1), symbols (1), icon (1), icons (1), pictures (0.8), picture font (1), arrows (0.8), borders (0.8), frames (0.8), pictogram (1), pictograms (1), clip art (1), clipart (1), silhouettes (0.8), fleuron (1), fleurons (1), printer ornaments (1) |  |
| fat-face | 0.905 | fat-face (49) | fat face (1), fatface (1) | ultra-bold, high-contrast |
| chancery | 0.964 | chancery (35) | chancery (1), chancery italic (1), italic hand (0.8) | calligraphy |
| modern-calligraphy | 0.989 | modern-calligraphy (47) | modern calligraphy (1), brush calligraphy (0.8) | calligraphy |
| dot-matrix | 0.935 | dot-matrix (31) | dot matrix (1), dotmatrix (1), led (0.6), receipt (0.6) |  |
| ocr | 0.812 | ocr (42) | ocr (1), machine readable (1), ocr a (1), ocr b (1) |  |

## weight

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| bold | 0.772 | bold (1517), heavy (1344), black (551), thick (87) | bold (1), heavy (1), black (0.6), thick (1), heavyweight (1) |  |
| ultra-bold | 0.824 | ultra-bold (117), ultra-black (56), fat (326), chunky (100), plump (73), ultra (81), extra-bold (26) | ultra bold (1), extra bold (1), extra-bold (1), ultra black (1), fat (1), chunky (1), plump (1), very bold (1), super bold (1) | bold |
| thin | 0.842 | thin (585), light (556), hairline (141) | thin (1), light (0.8), hairline (1), lightweight (1), fine (0.6), skinny strokes (0.8) |  |
| monoline | 0.869 | monoline (707), low-contrast (74), linear (302), monolinear (33) | monoline (1), low contrast (1), even stroke (1), uniform stroke (1), single line (0.8), monolinear (1) |  |
| high-contrast | 0.903 | high-contrast (245), contrast (170), thick-and-thin (57) | high contrast (1), thick and thin (1), thick thin (1), contrasty (0.8) |  |
| reverse-contrast | 0.976 | reverse-contrast (48) | reverse contrast (1), reversed contrast (1), reverse stress (1) |  |

## width

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| condensed | 0.887 | condense (798), narrow (930), compress (95), tall (165), skinny (63), ultra-narrow (23) | condensed (1), narrow (1), compressed (1), tall (0.8), skinny (0.8), tight (0.5), ultra narrow (1), extra condensed (1), compact (0.6) |  |
| wide | 0.840 | wide (547), extend (139), expand (37) | wide (1), extended (1), expanded (1), wide letters (1), stretched (0.6) |  |

## case

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| all-caps | 0.721 | all-caps (696), caps-only (311), uppercase (56), cap (345) | all caps (1), all-caps (1), caps only (1), uppercase (1), upper case (1), capitals (0.8), caps (0.8) |  |
| small-caps | 0.781 | small-caps (539) | small caps (1), small-caps (1) |  |
| lowercase | 0.780 | lowercase (48) | lowercase (1), lower case (1), all lowercase (1), no capitals (1) |  |

## slant

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| italic | 0.819 | italic (471), true-italics (222), slant (131), oblique (89), true-italic (48) | italic (1), italics (1), slanted (1), slant (1), oblique (1), leaning (0.8), tilted (0.6) |  |

## proportion

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| large-x-height | 0.801 | large-x-height (66), high-x-height (55) | large x height (1), high x height (1), big x height (1) |  |

## shape

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| rounded | 0.834 | round (1301), soft (606) | rounded (1), round (1), rounded corners (1), soft (0.8), bubble (0.6) |  |
| square | 0.870 | square (627), squarish (182), rectangular (112), block (471), octagonal (104), polygonal (78), chamfer (67), blocky (36), boxy (24), eurostile (28) | square (1), squared (1), squarish (1), boxy (1), blocky (1), block (0.8), rectangular (1), octagonal (1), chamfered (1), eurostile (0.8) |  |
| angular | 0.714 | angular (291), sharp (364), triangle (57) | angular (1), sharp (1), pointy (1), pointed (0.8), spiky (0.8), jagged (0.8), triangular (0.8) |  |
| modular | 0.823 | modular (290), grid (132), construct (143) | modular (1), grid (0.8), constructed (0.8) |  |
| swash | 0.846 | swash (1054), flourish (307), swirl (103), curl (80), curly (390), swash-caps (57), swirly (25) | swash (1), swashes (1), flourish (1), flourishes (1), flourished (1), swirly (1), swirls (1), curly (1), curls (0.8), loopy (0.6) |  |
| irregular | 0.805 | irregular (404), uneven (144), messy (117), random (106), jumpy (51), bouncy (320), loose (80), free-form (98), jag (79), wobbly (30) | irregular (1), uneven (1), messy (1), wobbly (1), bouncy (1), jumpy (1), loose (0.8), wonky (0.8), imperfect (0.8), shaky (0.8) |  |
| initials | 0.847 | initial (174), monogram (104) | initials (1), initial caps (1), drop cap (1), drop caps (1), monogram (1) |  |
| primitive | 0.753 | primitive (112) | primitive (1), crude (0.8), tribal (0.5) |  |
| bubble | 0.867 | bubble (44), bubbly (21), bulbous (31) | bubble (1), bubbles (1), bubbly (1), bulbous (1), puffy (0.8), balloon (0.8), inflated (0.8) |  |
| ink-traps | 0.803 | ink-traps (42) | ink traps (1), ink trap (1), inktraps (1) |  |

## texture

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| distressed | 0.850 | distress (372), rough (1069), erode (151), damage (115), decay (69), corrode (65), scratch (70), wear (117), noisy (78), texture (274), textured (182) | distressed (1), rough (1), worn (1), weathered (1), eroded (1), damaged (1), scratched (1), textured (1), texture (0.8), rusty (0.8), aged (0.6) |  |
| grunge | 0.866 | grunge (907), dirty (85), splatter (66), punk (211), destroy (33), destruct (47), trash (30) | grunge (1), grungy (1), dirty (1), splatter (1), splattered (1), punk (0.8), destroyed (1), trashed (0.8), wrecked (0.8) |  |
| letterpress | 0.754 | letterpress (266), stamp (156), rubber-stamp (23) | letterpress (1), stamp (1), stamped (1), rubber stamp (1) |  |
| wood-type | 0.830 | wood-type (428), wood (96), tuscan (138), circus (163) | wood type (1), woodtype (1), wood (0.6), tuscan (1), circus (0.8), carnival (0.6) |  |
| stitched | 0.863 | stitch (37), needlework (37) | stitch (1), stitched (1), stitching (1), embroidery (1), embroidered (1), cross stitch (1), needlework (1), sewing (0.8), sewn (0.8) |  |
| watercolor | 0.957 | watercolor (31) | watercolor (1), watercolour (1) |  |
| crayon | 0.903 | crayon (24) | crayon (1), crayons (1), wax crayon (1) |  |
| spray-paint | 0.799 | spray-paint (23), spray (28) | spray paint (1), spray painted (1), spraypaint (1), aerosol (1), spray (0.8) |  |
| spatter | 0.799 | spatter (24) | spatter (1), spattered (1), ink splatter (0.8) |  |

## effect

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| outline | 0.774 | outline (448) | outline (1), outlined (1), hollow (0.8), open (0.5) |  |
| inline | 0.828 | inline (331), prismatic (55) | inline (1), prismatic (1), striped (0.6) |  |
| shadow | 0.704 | shadow (248), shade (109) | shadow (1), shadowed (1), drop shadow (1), shaded (1) |  |
| 3d | 0.757 | 3d (277) | 3d (1), 3-d (1), three dimensional (1), dimensional (0.8) |  |
| layered | 0.745 | layer (241), chromatic (200) | layered (1), chromatic (1), multicolor (0.8), color font (0.6) |  |
| stencil | 0.839 | stencil (517) | stencil (1), stenciled (1), stencilled (1) |  |
| dotted | 0.836 | dot (143) | dotted (1), dots (1), polka dot (0.6) |  |
| neon | 0.907 | neon (109), tube (24) | neon (1), neon sign (1), glowing (0.6), neon tube (1) |  |
| distorted | 0.847 | distort (75) | distorted (1), warped (0.8), glitch (0.6), glitchy (0.6) |  |
| cut-out | 0.800 | cutup (64), scrap (68), cut-out (45) | cut out (1), cutout (1), ransom note (1), collage (0.8), cut up (1) |  |
| engraved | 0.825 | engrave (326) | engraved (1), engraving (1), etched (0.8) |  |
| chrome | 0.901 | chrome (23) | chrome (1), metallic (0.8), shiny (0.6) |  |
| marquee | 0.917 | marquee (31) | marquee (1), light bulbs (1), marquee lights (1), broadway (0.6) |  |
| striped | 0.875 | multi-line (44), stripe (46), strip (38) | striped (1), stripes (1), stripe (1), multi line (1), multiline (1), parallel lines (1) |  |

## decoration

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| ornate | 0.773 | ornate (192), ornamental (355), ornament (1006), embellishment (79), decoration (80) | ornate (1), ornamental (1), ornamented (1), embellished (1), decorated (0.8), fancy (0.6), decorative (0.6) |  |
| floral | 0.890 | flower (132), floral (85), botanical (95) | floral (1), flower (1), flowers (1), botanical (1), leaves (0.8), vines (0.8) |  |
| patterned | 0.871 | pattern (197) | patterned (1), pattern (1) |  |
| ribbon | 0.797 | ribbon (99), streamer (21) | ribbon (1), ribbons (1), folded ribbon (1), streamer (0.8) |  |

## era

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| art-deco | 0.793 | art-deco (890), artdeco (70), deco (276) | art deco (1), art-deco (1), artdeco (1), deco (1), gatsby (0.8), roaring twenties (0.6) |  |
| art-nouveau | 0.846 | art-nouveau (325), jugendstil (42) | art nouveau (1), nouveau (1), jugendstil (1) |  |
| bauhaus | 0.842 | bauhaus (199) | bauhaus (1) |  |
| modernist | 0.772 | modernism (64) | modernist (1), modernism (1), mid century (0.6), mid-century (0.6) |  |
| constructivist | 0.775 | constructivist (63), propaganda (52), soviet (30) | constructivist (1), constructivism (1), propaganda (1), soviet (1), ussr (0.8) |  |
| avant-garde | 0.756 | avant-garde (140), experimental (323) | avant garde (1), avant-garde (1), experimental (1) |  |
| victorian | 0.794 | victorian (189), 1800s (488), 1890s (58), 1880s (24) | victorian (1), 1800s (1), 19th century (1), 1890s (1), steampunk (0.6), 1880s (1) |  |
| early-1900s | 0.738 | 1900s (309), 1910s (77), edwardian (41) | 1900s (1), 1910s (1), early 1900s (1), edwardian (0.8) |  |
| 1920s | 0.818 | 1920s (451) | 1920s (1), 20s (0.8), twenties (0.8) |  |
| 1930s | 0.804 | 1930s (585) | 1930s (1), 30s (0.8), thirties (0.8) |  |
| 1940s | 0.782 | 1940s (397) | 1940s (1), 40s (0.8), forties (0.8), wartime (0.6) |  |
| 1950s | 0.750 | 1950s (484) | 1950s (1), 50s (0.8), fifties (0.8) |  |
| 1960s | 0.719 | 1960s (455) | 1960s (1), 60s (0.8), sixties (0.8) |  |
| 1970s | 0.787 | 1970s (378) | 1970s (1), 70s (0.8), seventies (0.8) |  |
| 1980s | 0.669 | 1980s (230) | 1980s (1), 80s (0.8), eighties (0.8) |  |
| 18th-century | 0.945 | 1700s (92) | 1700s (1), 18th century (1), colonial (0.6) |  |
| medieval | 0.836 | medieval (105), monastic (36), manuscript (21), incunabula (26) | medieval (1), middle ages (1), knight (0.5), monastic (0.8), illuminated manuscript (1), manuscript (0.8), incunabula (0.8) |  |
| baroque | 0.932 | baroque (64) | baroque (1), rococo (0.8) |  |
| historical | 0.766 | historical (127), historic (73), ancient (523), classical (56) | historical (1), historic (1), ancient (1), classical (0.8) |  |
| vintage | 0.719 | vintage (1601), antique (672), old (250), old-fashioned (80), nostalgic (57), quaint (142) | vintage (1), antique (1), old fashioned (1), old-fashioned (1), old (0.6), nostalgic (0.8), classic (0.5) |  |
| retro | 0.710 | retro (2319), revival (195) | retro (1), throwback (0.8), revival (0.6) |  |
| classic | 0.754 | classic (570), traditional (66), conservative (167) | classic (1), traditional (1), timeless (0.8), conservative (0.8) |  |
| western | 0.901 | western (260), wild-west (262), cowboy (94), rodeo (63), country (90) | western (1), wild west (1), cowboy (1), rodeo (1), saloon (0.8), wanted poster (0.8), country (0.6) |  |
| psychedelic | 0.874 | psychedelic (63), hippie (65), groovy (49) | psychedelic (1), trippy (1), hippie (1), groovy (0.8), flower power (0.8) |  |
| disco | 0.802 | disco (131) | disco (1), funk (0.6) |  |
| hipster | 0.805 | hipster (328) | hipster (1) |  |
| futuristic | 0.846 | futuristic (589), future (83), sci-fi (143), space (98), alien (53), robot (79), scifi (24), robotic (49) | futuristic (1), future (0.8), sci fi (1), sci-fi (1), scifi (1), rocket (0.6), outer space (1), science fiction (1), space (0.8), alien (0.8), robot (0.8), cyberpunk (0.6), robotic (1) |  |
| techno | 0.887 | techno (534), tech (60), technology (104), computer (440), hi-tech (34), high-tech (23) | techno (1), tech (1), technology (1), computer (1), cyber (0.8), hi tech (1), high tech (1) |  |
| technical | 0.810 | technical (565), mechanical (271), industrial (325), industry (83), architect (120), architecture (71), din (115), industrial-sans (40), machinery (29), mechanic (29), railroad (38) | technical (1), mechanical (1), industrial (1), engineering (0.8), din (1), architectural (0.8), blueprint (0.6), machinery (0.8), machine (0.8), railroad (0.6), railway (0.6) |  |
| signage | 0.715 | signage (915), wayfinding (64), information (184), transport (96), highway (23), traffic (42), metro (32) | signage (1), wayfinding (1), road sign (1), airport (0.6), highway (1), traffic sign (1), metro (0.8), subway (0.8), transit (0.8) |  |
| sign-painting | 0.855 | sign-painting (335), showcard (120) | sign painting (1), sign painter (1), signwriting (1), showcard (1) |  |
| graffiti | 0.826 | graffiti (191), grafitti (76), street (107) | graffiti (1), grafitti (1), street art (1), tag (0.5), spray paint (0.6) |  |
| urban | 0.705 | urban (302), hip-hop (49) | urban (1), street (0.6), hip hop (1), rap (0.8) |  |
| tattoo | 0.747 | tattoo (200), tatoo (22) | tattoo (1), tattoos (1) |  |
| rock | 0.777 | heavy-metal (67), metal (95), rock (75) | heavy metal (1), metal (0.8), rock (0.8), rock and roll (0.8), band (0.5) |  |
| streamline | 0.887 | streamline (47) | streamline (1), streamlined (1), streamline moderne (1) | art-deco |
| early-modern | 0.956 | 1500s (33), 1600s (39) | 1500s (1), 1600s (1), 16th century (1), 17th century (1), early modern (1) |  |

## theme

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| asian-style | 0.796 | asian (67), far-east (31) | asian (1), chinese (0.8), japanese (0.8), oriental (0.6), chopstick (0.8), far east (1), bamboo (0.8), asian style (1) |  |
| arabic-style | 0.849 | arabic (93) | arabic (1), middle eastern (0.8), arabesque (0.6), arabic style (1) |  |
| faux-cyrillic | 0.748 | russian (168), old-russian (44) | russian (1), faux cyrillic (1), old russian (1) |  |
| mexican | 0.803 | mexican (56) | mexican (1), fiesta (0.6) |  |
| african | 0.840 | africa (58) | african (1), africa (1), tribal (0.6) |  |
| horror | 0.821 | horror (179), scary (85), evil (71), monster (98), ghost (131), dark (80), spooky (45), creepy (27), vampire (37), death (44), skull (24), skeleton (21), blood (33) | horror (1), scary (1), spooky (1), creepy (1), evil (1), monster (1), ghost (1), zombie (0.8), blood (0.8), bloody (0.8), dark (0.6), gothic horror (1), vampire (1), skull (0.8), skeleton (0.8), death (0.8) |  |
| halloween | 0.868 | halloween (218), witch (29) | halloween (1), witch (0.8), witches (0.8) | horror |
| fantasy | 0.748 | fantasy (83), magic (105), mysterious (75), wizard (32) | fantasy (1), magic (1), mystical (1), mysterious (0.8), wizard (0.8) |  |
| pirate | 0.891 | pirate (69) | pirate (1), nautical (0.6) |  |
| christmas | 0.768 | christmas (239), xmas (227), winter (53), snow (24), snowflake (26) | christmas (1), xmas (1), holiday (0.6), winter (0.8), snow (0.8), festive (0.6), snowflake (0.8), snowy (0.8) |  |
| valentine | 0.761 | valentine (415), love (339), heart (108) | valentine (1), valentines (1), love (0.8), hearts (0.8) |  |
| party | 0.733 | party (414), birthday (140), celebration (121), festive (142), holiday (157) | party (1), birthday (1), celebration (1), festive (0.8), holiday (0.8) |  |
| wedding | 0.857 | wed (1045) | wedding (1), weddings (1), bridal (1), bride (0.8), save the date (1), anniversary (0.6) |  |
| invitation | 0.802 | invitation (1272), invite (77), announcement (76), stationery (196), stationary (96), greeting-card (139), greet (382), card (186), postcard (122) | invitation (1), invitations (1), invite (1), stationery (1), greeting card (1), card (0.6), postcard (0.8), announcement (0.8) |  |
| certificate | 0.767 | certificate (192), diploma (88) | certificate (1), diploma (1), award (0.6) |  |
| kids | 0.816 | kid (793), child (587), childish (74), childrens-book (64), baby (53), toy (123) | kids (1), kid (1), children (1), childrens (1), child (1), childish (1), childlike (1), baby (0.8), toddler (0.8), toy (0.8), storybook (0.8), nursery (0.8) |  |
| school | 0.769 | school (238), education (104) | school (1), education (1), classroom (1), teacher (0.8) |  |
| comic | 0.843 | comic (910), comic-book (147), comic-text (175), cartoon (480), animate (91), animation (52) | comic (1), comics (1), comic book (1), cartoon (1), cartoony (1), animated (0.8), animation (0.8), manga (0.6), speech bubble (0.8) |  |
| video-game | 0.842 | video-game (127) | video game (1), videogame (1), gaming (0.8), arcade (0.8), game (0.6) |  |
| sport | 0.744 | sport (350), athletic (53), football (62), basketball (40), hockey (30) | sport (1), sports (1), sporty (1), athletic (1), football (0.8), jersey (0.8), basketball (1), hockey (1) |  |
| varsity | 0.798 | college (70) | varsity (1), collegiate (1), college (1) |  |
| baseball-script | 0.915 | baseball-script (52), baseball (65) | baseball (1), baseball script (1) | script, sport |
| speed | 0.771 | fast (223), speed (62) | speed (1), fast (1), racing (1), motion (0.6) |  |
| military | 0.869 | military (169) | military (1), army (1) |  |
| surf | 0.796 | surf (123), surfer (63) | surf (1), surfing (1), surfer (1) |  |
| tropical | 0.752 | tropical (68), beach (78), tiki (48) | tropical (1), beach (1), hawaiian (0.8), tiki (1) |  |
| summer | 0.721 | summer (170), vacation (189) | summer (1), vacation (0.8) |  |
| skateboard | 0.958 | skatebord (53) | skateboard (1), skate (1), skater (0.8) |  |
| menu | 0.725 | menu (327), restaurant (180), cafe (66), coffee (99), bakery (55), food (528) | menu (1), restaurant (1), cafe (1), coffee (1), bakery (1), food (0.8), recipe (0.6), diner (0.6), gourmet (0.6) |  |
| candy | 0.842 | candy (152) | candy (1), sweets (0.8) |  |
| corporate | 0.842 | corporate (833), business (188), business-text (113), office (100), identity (208), corporative (21), professional (46) | corporate (1), business (1), professional (0.8), office (0.8) |  |
| logo | 0.724 | logo (1857), logotype (648), brand (1356) | logo (1), logos (1), logotype (1), branding (1), brand (1) |  |
| newspaper | 0.807 | news (628), newspaper (369), news-headline (148), editorial (892), newsletter (164) | newspaper (1), news (1), editorial (1), newsletter (0.8), headline (0.5) |  |
| magazine | 0.783 | magazine (2740), magazin (36) | magazine (1) |  |
| text | 0.857 | text (1315), book-text (175), news-text (88), workhorse (360), small-text (60), functional (97), book (922) | body text (1), text (0.8), book (0.8), reading (0.8), long text (1), paragraph (0.8), workhorse (1) |  |
| legible | 0.786 | legible (2504), readable (204), clear (234) | legible (1), readable (1), easy to read (1), clear (0.8), readability (1) |  |
| display | 0.681 | display (4233), headline (4129), title (652), poster (3489), header (99), billboard (141), advertise (1211), flyer (162) | display (1), headline (1), headlines (1), title (0.8), titles (0.8), poster (0.8), posters (0.8), heading (0.8), billboard (0.8), advertising (0.6) |  |
| decorative | 0.679 | decorative (5059), fancy (907), novelty (225) | decorative (1), fancy (1), novelty (1) |  |
| fashion | 0.723 | fashion (963), fashionable (810), cosmetic (142), beauty (94), perfume (117), glamour (51), jewelry (69), chic (104), perfums (26) | fashion (1), fashionable (1), beauty (1), cosmetics (1), perfume (1), glamour (1), glamorous (1), jewelry (1), chic (1), luxury (0.6), vogue (0.8) |  |
| user-interface | 0.910 | ui (44), user-interface (33), interface (41), apps (47) | ui (1), user interface (1), interface (1), app (0.8), apps (0.8), dashboard (0.6) |  |
| celtic | 0.959 | celtic (33) | celtic (1), irish (0.6), gaelic (0.8), insular (0.8) |  |
| runic | 0.881 | rune (28) | rune (1), runes (1), runic (1), viking (0.6), norse (0.6) |  |
| occult | 0.887 | occult (23) | occult (1), witchcraft (0.8), tarot (0.8), esoteric (0.8), mystic (0.6) |  |
| fairy-tale | 0.781 | fairytale (92), fairy (24), fairy-tale (35), tale (43), magical (20) | fairy tale (1), fairytale (1), fairy (1), fairies (1), enchanted (0.8), magical (0.8) |  |
| folk | 0.829 | folk (37) | folk (1), folk art (1), folksy (0.8) |  |

## mood

| canonical | AUC | members (train fonts) | aliases | implies |
|---|---|---|---|---|
| elegant | 0.812 | elegant (2207), graceful (269), beautiful (291), classy (65), sophisticate (171), refine (58), delicate (358) | elegant (1), elegance (1), graceful (1), classy (1), sophisticated (1), refined (1), delicate (0.8), beautiful (0.8), luxury (0.6), luxurious (0.6), upscale (0.6) |  |
| formal | 0.838 | formal (616), royal (76) | formal (1), royal (0.8), regal (0.8), stately (0.8) |  |
| romantic | 0.783 | romantic (429) | romantic (1), romance (1), sensual (0.6) |  |
| feminine | 0.773 | feminine (732), girly (196), girl (90), woman (69) | feminine (1), girly (1), girlish (1), girl (0.8), womanly (1), female (0.8) |  |
| masculine | 0.762 | masculine (421) | masculine (1), manly (1), male (0.8) |  |
| rugged | 0.722 | rugged (75), tough (76), sturdy (203), robust (71) | rugged (1), tough (1), sturdy (1), robust (1), strong (0.6) |  |
| cute | 0.768 | cute (814), sweet (231), lovely (187), pretty (74), charm (54), adorable (29) | cute (1), sweet (1), adorable (1), lovely (0.8), pretty (0.8), charming (0.8), kawaii (1) |  |
| playful | 0.736 | playful (624), fun (1167), whimsical (136), lively (412), happy (386), silly (60) | playful (1), fun (1), whimsical (1), lively (1), happy (1), cheerful (1), silly (1), joyful (1) |  |
| funny | 0.765 | funny (1691), crazy (318), wild (319), comedy (33) | funny (1), humorous (1), goofy (1), crazy (0.8), wacky (0.8), zany (0.8), comedy (1), comedic (1) |  |
| quirky | 0.703 | quirky (223), eccentric (64), unusual (483), weird (77), idiosyncratic (122), funky (191), offbeat (36) | quirky (1), eccentric (1), unusual (1), weird (1), odd (0.8), funky (1), strange (0.8), offbeat (1) |  |
| friendly | 0.739 | friendly (1157), warm (110), personable (64) | friendly (1), warm (1), approachable (1), welcoming (0.8), inviting (0.8) |  |
| casual | 0.788 | casual (1024), informal (1950), freestyle (67) | casual (1), informal (1), relaxed (0.8), laid back (0.8), easygoing (0.8) |  |
| modern | 0.729 | modern (2567), contemporary (1527), fresh (362), trendy (126), stylish (578) | modern (1), contemporary (1), trendy (1), stylish (0.8), fresh (0.8), current (0.6) |  |
| clean | 0.803 | clean (1519), simple (531), plain (206), basic (73), neat (80), crisp (77), minimal (307), neutral (323), modest (124), minimalist (29) | clean (1), simple (1), minimal (1), minimalist (1), minimalistic (1), plain (1), basic (0.8), neat (0.8), crisp (0.8), neutral (0.8), understated (0.8) |  |
| organic | 0.745 | organic (776), natural (347), nature (114) | organic (1), natural (1), nature (0.8), earthy (0.8), eco (0.6) |  |
| rustic | 0.810 | rustic (221) | rustic (1), farmhouse (0.8), handcrafted (0.5) |  |
| edgy | 0.672 | edgy (103), attitude (53) | edgy (1), attitude (0.8), rebellious (0.8), aggressive (0.8) |  |

## Dropped

- **language/character-set support, not visible in latin glyphs**: accent, american, capital-sharp-s, cyrillic, czech, diacritic, dutch, english, european, french, german, greek, italian, latin, latino, multilingual, spanish, turkish, ukrainian, versal-eszett
- **OpenType feature or font-product metadata**: alternate, ball-terminals, collection, contextual-alternates, counterless, extra, family, filled-counters, fraction, free, kern, ligature, linotype-taketype, median-spurs, new, old-style-numerals, oldstyle-figures, opentype, optical-sizes, personal, personalize, pintassilgoprints, pro, regular, spur-serif, spurless, static, stylistic-alternates, superfamily, tail, type, typeface, typography, unicase, want, webfont, wishlist
- **usage context with low visual consistency (AUC < 0.72 or grounding ~0)**: 1990s, 2000s, 2010s, abstract, ad, alternative, angle, animal, app, art, artistic, arts-and-crafts, artsy, authentic, banner, big, blog, book-cover, break, brochure, capital, caption, car, catchword, church, circle, color, commercial, construction, cool, correspondence, cover, craft, creative, curve, curvy, dance, design, disconnect, distinctive, diy, dynamic, easy, economic, emblem, ethnic, exotic, expressive, extreme, fill, film, flair, flash, flow, fluid, gaspipe, graphic, green, head, hybrid, illustration, impact, ink, interlock, jazz, label, letter, letterhead, line, luxury, manual, market, masthead, movie, movie-credits, music, nightclub, nightlife, open, original, package, package-design, poetry, point, press, print, product, product-packaging, publication, publish, quick, quote, sassy, scrapbook, screen, sensible, sexy, sign, smooth, solid, star, strong, style, t-shirt, tag, television, toolkit, travel, tv, unique, upright, useful, valuable, versatile, video, water, weather, web, web-graphics, website, wine, young, youth
- **ambiguous across unrelated styles (blackletter vs. gothic sans vs. horror); 'gothic' aliases to blackletter at 0.8**: gothic
- **merged into a canonical it lowered the AUC of (validate_tag_vocabulary.py); searchable only via aliases**: cut, festive-occasions, game
- **family-specific (AUC 0.96 from one or two families), not a style**: realist
- **20-49 band: language, place or character-set support**: %d0%ba%d0%b8%d1%80%d0%b8%d0%bb%d0%bb%d0%b8%d1%86%d0%b0, america, british, bulgarian, central-europe, chile, chilean, chinese, cuba, cyr, eszett, italy, japan, latin-american, paris, persian, polish, romanian, slovak, urdu, vietnamese
- **20-49 band: foundry, family or product metadata (or >=50% one foundry prefix)**: bi-form, bluemlein, bluemlein-script-collection, bundle, century, companion, creamy, dandy, daniel-hernandez, element, fav, favorite, feature, generic, glyph, grayletter, latinotype, lead, letterbat, medium, motif, no-baseline, no-counter, normal, number, old-style-figures, opentype-features, pack, pap, personal-text, rsz, scrapper, seduce, software, standard, tabular, teacher, tech-pubs, universal, viergutz
- **20-49 band: usage context or subject, not a visual style**: action, adventure, alcohol, anniversary, badge, bamboo, beer, bible, bird, body, booklet, cartography, catalogue, character, chocolate, cocktail, compact, cross, diary, diner, dinner-invitation, drop, economical, fax, fiction, funeral, gangster, gift, gourmet, halftone, hands-on, history, infographic, math, meal, memo, money, people, period, plant, printer, recipe, retail, reverse, save-the-date, science, scrapbooking, shop, sixty, spring, sun, tea, teen, teenage, teenager, textile, tool, tree, vector, vignette, zodiac
- **20-49 band: mood or quality word too vague to search as a tag**: agile, alive, boho, bounce, coarse, expressionist, eye-catcher, fanciful, feather, fine, hand-cut, hand-painted, human, informal-text, inky, large-aperture, loop, loud, mono, movement, personality, raw, regal, relax, rigid, rocky, sensationalist, sensual, serious, shabby-chic, slim, spiral, spontaneous, upright-italic, vernacular, wire
- **20-49 band: tagger AUC < 0.75**: americana, architectural, army, attractive, awesome, baltic, bar, bevel, bizarre, blur, body-text, box, brazil, broadcast, canadian, carnival, carve, casino, cat, catalog, celebrate, cheap, childlike, childrens, chisel, cinema, clothe, communication, cook, custom, detail, device, diamond, discretionary-ligatures, dramatic, drop-shadow, e-book, e-pub, ebook, effect, energetic, energy, epub, exclusive, eye-catching, face, female, festival, fire, fish, fruit, funk, geology, geometry, goth, graphic-design, hint, hot, illustrative, incise, indian, indie, international, japanese, large-eye, leaf, legibility, machine, macho, manicule, metallic, mix, model, moderne, modernist, monumental, motorcycle, naive, nautical, navigation, nice, ocean, old-english, old-school, optical, paper, pop, popular, powerful, punch, read, realistic, reclame, science-fiction, screen-design, sea, shape, skate, skateboard, smart, space-age, spain, special, spiky, sporty, steampunk, straight, stroke, stylistic, stylize, system, tight, trend, ugly, underline, usa, varsity, vegetable, wavy, weight, woodcut, zombie
