"""
Visual sanity check for experiments/embedding-geometry/ssim_duplicate_pairs.csv: scrolls through every
candidate pair, weakest (most borderline) aligned-SSIM score first, so you can eyeball whether the
pipeline's low end is actually catching real near-duplicates or garbage, then scroll down toward the
obviously-identical high end. Shows both glyphs ('a' and 'g') for each font in the pair, the aligned
SSIM score, and each trained embedding's own cosine for the same pair (cos_all / cos_allText).

Controls: mouse wheel to scroll, drag the scrollbar on the right, PageUp/PageDown, Home/End.

    python experiments/embedding-geometry/browse_ssim_pairs.py [csvPath]
"""
import os as _os
import sys as _sys
_scriptDir = _os.path.dirname(_os.path.abspath(__file__))
_repoRoot = _os.path.dirname(_os.path.dirname(_scriptDir))
if _repoRoot not in _sys.path:
    _sys.path.insert(0, _repoRoot)
_os.chdir(_repoRoot)

import csv
import pickle

import cv2
import numpy as np
import pygame

CSV_PATH = _sys.argv[1] if len(_sys.argv) > 1 else _os.path.join("experiments", "embedding-geometry", "ssim_duplicate_pairs.csv")
GLYPHS = ["a", "g"]

WIDTH, HEIGHT = 1180, 800
ROW_HEIGHT = 130
THUMB = 96
SCROLLBAR_WIDTH = 18
BG = (24, 24, 28)
ROW_BG_A = (34, 34, 40)
ROW_BG_B = (30, 30, 36)
TEXT_COLOR = (230, 230, 230)
DIM_COLOR = (150, 150, 160)
SCORE_LOW = (220, 90, 90)
SCORE_MID = (220, 190, 90)
SCORE_HIGH = (110, 210, 130)
SCROLLBAR_BG = (45, 45, 52)
SCROLLBAR_HANDLE = (100, 100, 115)


def loadBitmap(path):
    try:
        img = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        return img.astype(np.float64) / 255.0 if img is not None else None
    except Exception:
        return None


def toSurface(img, size=THUMB):
    if img is None:
        surf = pygame.Surface((size, size))
        surf.fill((60, 20, 20))
        return surf
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1).transpose(1, 0, 2)  # (W,H,3) for pygame surfarray
    surf = pygame.surfarray.make_surface(rgb)
    return pygame.transform.smoothscale(surf, (size, size))


def scoreColor(score):
    if score is None:
        return DIM_COLOR
    if score < 0.85:
        return SCORE_LOW
    if score < 0.95:
        return SCORE_MID
    return SCORE_HIGH


def truncate(font, text, maxWidth):
    if font.size(text)[0] <= maxWidth:
        return text
    while text and font.size(text + "...")[0] > maxWidth:
        text = text[:-1]
    return text + "..."


class Cache:
    """Bounded LRU-ish surface cache -- rows get re-rendered as you scroll past them repeatedly."""

    def __init__(self, maxSize=1500):
        self.maxSize = maxSize
        self.data = {}
        self.order = []

    def get(self, key, build):
        if key in self.data:
            self.order.remove(key)
            self.order.append(key)
            return self.data[key]
        value = build()
        self.data[key] = value
        self.order.append(key)
        if len(self.order) > self.maxSize:
            old = self.order.pop(0)
            del self.data[old]
        return value


def main():
    with open("embeddings/fontGlyphPaths.pkl", "rb") as f:
        pathMap = pickle.load(f)

    rows = []
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["alignedSsim"] = float(r["alignedSsim"]) if r["alignedSsim"] else None
                r["cos_all"] = float(r["cos_all"]) if r.get("cos_all") else None
                r["cos_allText"] = float(r["cos_allText"]) if r.get("cos_allText") else None
            except ValueError:
                continue
            if r["alignedSsim"] is not None:
                rows.append(r)
    rows.sort(key=lambda r: r["alignedSsim"])  # lowest (most borderline) first
    print(f"{len(rows)} pairs loaded from {CSV_PATH}, sorted lowest aligned-SSIM first", flush=True)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("SSIM duplicate pairs -- weakest first, scroll down for stronger matches")
    clock = pygame.time.Clock()
    fontBig = pygame.font.SysFont("consolas", 18)
    fontSmall = pygame.font.SysFont("consolas", 14)

    cache = Cache()
    scroll = 0  # in rows (float, for smooth wheel scroll)
    maxScroll = max(0, len(rows) - 1)
    dragging = False

    def glyphSurface(fontData, letter):
        path = fontData.get(letter) if fontData else None
        if not path:
            return toSurface(None)
        return cache.get(("img", path), lambda: toSurface(loadBitmap(path)))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                scroll -= event.y * 2
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PAGEDOWN:
                    scroll += HEIGHT / ROW_HEIGHT
                elif event.key == pygame.K_PAGEUP:
                    scroll -= HEIGHT / ROW_HEIGHT
                elif event.key == pygame.K_HOME:
                    scroll = 0
                elif event.key == pygame.K_END:
                    scroll = maxScroll
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and event.pos[0] >= WIDTH - SCROLLBAR_WIDTH:
                    dragging = True
                elif event.button == 4:
                    scroll -= 2
                elif event.button == 5:
                    scroll += 2
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                frac = max(0.0, min(1.0, event.pos[1] / HEIGHT))
                scroll = frac * maxScroll

        scroll = max(0, min(maxScroll, scroll))
        screen.fill(BG)

        startIdx = int(scroll)
        yOffset = -(scroll - startIdx) * ROW_HEIGHT
        y = yOffset
        i = startIdx
        while y < HEIGHT and i < len(rows):
            r = rows[i]
            rowRect = pygame.Rect(0, y, WIDTH - SCROLLBAR_WIDTH, ROW_HEIGHT)
            pygame.draw.rect(screen, ROW_BG_A if i % 2 == 0 else ROW_BG_B, rowRect)

            fa = pathMap.get(r["fontA"], {})
            fb = pathMap.get(r["fontB"], {})
            x = 12
            for g in GLYPHS:
                screen.blit(glyphSurface(fa, g), (x, y + (ROW_HEIGHT - THUMB) // 2))
                x += THUMB + 6

            nameW = 260
            nameText = truncate(fontBig, r["fontA"], nameW)
            screen.blit(fontBig.render(nameText, True, TEXT_COLOR), (x, y + 14))
            screen.blit(fontSmall.render(f"cos_all={r['cos_all']:.3f}" if r["cos_all"] is not None else "cos_all=?",
                                          True, DIM_COLOR), (x, y + 40))
            screen.blit(fontSmall.render(f"cos_allText={r['cos_allText']:.3f}" if r["cos_allText"] is not None else "cos_allText=?",
                                          True, DIM_COLOR), (x, y + 58))
            x += nameW + 8

            scoreText = f"{r['alignedSsim']:.4f}"
            scoreSurf = fontBig.render(scoreText, True, scoreColor(r["alignedSsim"]))
            screen.blit(scoreSurf, (x, y + ROW_HEIGHT // 2 - 10))
            x += 90

            nameText = truncate(fontBig, r["fontB"], nameW)
            screen.blit(fontBig.render(nameText, True, TEXT_COLOR), (x, y + 14))
            x += nameW + 8

            for g in GLYPHS:
                screen.blit(glyphSurface(fb, g), (x, y + (ROW_HEIGHT - THUMB) // 2))
                x += THUMB + 6

            y += ROW_HEIGHT
            i += 1

        # scrollbar
        pygame.draw.rect(screen, SCROLLBAR_BG, (WIDTH - SCROLLBAR_WIDTH, 0, SCROLLBAR_WIDTH, HEIGHT))
        if maxScroll > 0:
            handleH = max(30, HEIGHT * (HEIGHT / ROW_HEIGHT) / len(rows))
            handleY = (scroll / maxScroll) * (HEIGHT - handleH)
            pygame.draw.rect(screen, SCROLLBAR_HANDLE, (WIDTH - SCROLLBAR_WIDTH + 2, handleY, SCROLLBAR_WIDTH - 4, handleH), border_radius=4)

        header = f"{int(scroll) + 1}/{len(rows)}  (weakest score at top, scroll down for stronger matches)"
        screen.blit(fontSmall.render(header, True, DIM_COLOR), (10, HEIGHT - 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
