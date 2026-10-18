"""Feature 2: search inside a slide image for a phrase and draw a heatmap.

Steps: cut the image into tiles -> skip empty glass -> score each tile -> color
the tiles by score on top of the image.

How tiles are scored (real model): each tile is compared with EVERY term in
prompts/phrases.yaml at once, and the score is the chance (0 to 1) that the tile
is the chosen term rather than one of the others. This is CONCH's zero-shot
classification, applied tile by tile. Because the score is relative to the other
terms, it is comparable across phrases, and a term that isn't in the image scores
low everywhere instead of always lighting something up. Colors use a fixed
0-100% scale for the same reason.

Uses the team's code: load_conch() and encode_text() from src/models.py, and the
phrases + templates in prompts/phrases.yaml (templates use {} for the phrase).

Develop on a laptop with the placeholder scorer (no model needed):
    python -m src.heatmap --image test_images/he.jpg --phrase necrosis --fake
With CONCH:
    python -m src.heatmap --image test_images/he.jpg --phrase necrosis
"""

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATES = ["an H&E image of {}."]
# Used only if phrases.yaml has fewer than two terms, so there is always something to compare against
FALLBACK_PANEL = ["tumor cells", "stroma", "normal tissue", "necrosis", "lymphocytes", "adipose tissue"]


def load_phrases(path=REPO_ROOT / "prompts" / "phrases.yaml"):
    """Return {"templates": [...], "phrases": {name: [synonyms]}}; tolerates a missing or empty file."""
    cfg = {}
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    return {
        "templates": cfg.get("templates") or DEFAULT_TEMPLATES,
        "phrases": cfg.get("phrases") or {},
    }


def phrase_texts(phrase, cfg):
    """All sentences for one term: every synonym dropped into every template.
    A term not in phrases.yaml (typed by the user) is used as its own only synonym."""
    synonyms = cfg["phrases"].get(phrase, [phrase])
    return [t.format(s) for s in synonyms for t in cfg["templates"]]


# ---------- Tiling ----------

def tile_image(img, tile_size=256):
    """Cut an image into non-overlapping square tiles (leftover edge pixels are dropped).

    Returns (tiles, coords, (n_rows, n_cols)), where coords[i] = (row, col) of tiles[i].
    """
    w, h = img.size
    n_cols, n_rows = w // tile_size, h // tile_size
    tiles, coords = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            box = (c * tile_size, r * tile_size, (c + 1) * tile_size, (r + 1) * tile_size)
            tiles.append(img.crop(box))
            coords.append((r, c))
    return tiles, coords, (n_rows, n_cols)


def is_tissue(tile, min_stained=0.01):
    """Keep a tile if at least 1% of its pixels are stained (colored or dark).

    Empty glass is near-white everywhere. Tissue that is mostly white space, like fat
    or lung, still has thin stained walls, so it passes; the old average-color check
    threw those tiles away.
    """
    hsv = np.asarray(tile.convert("HSV"), dtype=np.float32) / 255.0
    stained = (hsv[..., 1] > 0.12) | (hsv[..., 2] < 0.6)
    return float(stained.mean()) >= min_stained


# ---------- Scorers (same interface: .name and .score(tiles, phrase, cache_key=None) -> values 0..1) ----------

class FakeScorer:
    """Placeholder for development only. Ignores the phrase and scores tiles by how dark
    the stain is (dense nuclei look dark), so tiling and drawing can be built without a model."""

    name = "fake"

    def score(self, tiles, phrase, cache_key=None):
        return np.array([1 - np.asarray(t.convert("L"), dtype=np.float32).mean() / 255 for t in tiles])


class ConchScorer:
    """Real scorer: per tile, the chance (0-1) that it is `phrase` rather than any other term in phrases.yaml."""

    name = "conch"

    def __init__(self, checkpoint_path=None, batch_size=64):
        from src.models import load_conch

        # checkpoint_path=None uses conch_checkpoint from configs/paths.yaml
        cfg = {"conch_checkpoint": checkpoint_path} if checkpoint_path else None
        self.model, self.preprocess, self.tokenizer = load_conch(cfg)
        self.batch_size = batch_size
        self.cfg = load_phrases()
        scale = getattr(self.model, "logit_scale", None)
        self.logit_scale = float(scale.detach().exp()) if scale is not None else 100.0
        self._tile_cache = {}  # embedding tiles is the slow part, so reuse it across phrases
        self._text_cache = {}

    def _panel(self, phrase):
        """The terms each tile is compared against: everything in phrases.yaml, plus a typed phrase."""
        names = list(self.cfg["phrases"])
        if len(names) < 2:
            names = list(dict.fromkeys(names + FALLBACK_PANEL))
        if phrase not in names:
            names.append(phrase)
        return names

    def _text_vector(self, phrase):
        """Average the embeddings of all sentences for a term (a prompt ensemble)."""
        if phrase not in self._text_cache:
            import torch.nn.functional as F

            from src.models import encode_text

            emb = encode_text(self.model, self.tokenizer, phrase_texts(phrase, self.cfg)).mean(0)
            self._text_cache[phrase] = F.normalize(emb, dim=-1)
        return self._text_cache[phrase]

    def _embed_tiles(self, tiles):
        import torch
        import torch.nn.functional as F

        from src.models import DEVICE

        out = []
        with torch.no_grad():
            for i in range(0, len(tiles), self.batch_size):
                x = torch.stack([self.preprocess(t) for t in tiles[i : i + self.batch_size]]).to(DEVICE)
                out.append(F.normalize(self.model.encode_image(x), dim=-1))
        return torch.cat(out)

    def score(self, tiles, phrase, cache_key=None):
        import torch

        if cache_key is not None and cache_key in self._tile_cache:
            tile_emb = self._tile_cache[cache_key]
        else:
            tile_emb = self._embed_tiles(tiles)
            if cache_key is not None:
                self._tile_cache[cache_key] = tile_emb

        names = self._panel(phrase)
        text = torch.stack([self._text_vector(n) for n in names])        # [terms, D]
        probs = (self.logit_scale * tile_emb @ text.T).softmax(dim=-1)    # [tiles, terms]
        return probs[:, names.index(phrase)].float().cpu().numpy()


# ---------- Heatmap ----------

def draw_overlay(img, grid, tile_size, max_alpha=0.6, cmap="viridis", vmin=0.0, vmax=1.0):
    """Color each tissue tile by its score on a FIXED scale (0 to 1 by default), so a weak
    match looks weak even if it is the best tile in the image. Higher scores are brighter
    and more opaque; skipped tiles (NaN) stay transparent. max_alpha=0 returns the tissue unchanged."""
    import matplotlib

    norm = np.clip(np.nan_to_num((grid - vmin) / (vmax - vmin)), 0.0, 1.0)
    colors = matplotlib.colormaps[cmap](norm)  # [rows, cols, 4], values 0-1
    colors[..., 3] = np.where(np.isnan(grid), 0.0, max_alpha * (0.08 + 0.92 * norm))
    heat = Image.fromarray((colors * 255).astype(np.uint8))
    heat = heat.resize((grid.shape[1] * tile_size, grid.shape[0] * tile_size), Image.NEAREST)

    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(heat, (0, 0))
    return Image.alpha_composite(base, layer).convert("RGB")


def heatmap_for_image(img, phrase, scorer, tile_size=256, min_stained=0.01, n_top=5):
    """Run the whole pipeline on one image.

    Returns (overlay_image, top_tiles, grid):
      overlay_image: the input with the heatmap drawn on top
      top_tiles:     [(tile_image, score), ...] for the n_top best-matching regions
      grid:          score per tile position, 0 to 1 (NaN = skipped as empty glass)
    """
    img = img.convert("RGB")
    tiles, coords, (n_rows, n_cols) = tile_image(img, tile_size)
    if not tiles:
        raise ValueError(f"Image is smaller than one {tile_size}px tile. Use a bigger image or a smaller tile size.")

    keep = [i for i, t in enumerate(tiles) if is_tissue(t, min_stained)]
    if not keep:
        raise ValueError("No tissue found. The image looks like empty glass.")

    cache_key = f"{hashlib.md5(img.tobytes()).hexdigest()}_{tile_size}_{min_stained}"
    scores = np.asarray(scorer.score([tiles[i] for i in keep], phrase, cache_key=cache_key), dtype=np.float32)

    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for i, s in zip(keep, scores):
        grid[coords[i]] = s

    overlay = draw_overlay(img, grid, tile_size)
    best = np.argsort(scores)[::-1][:n_top]
    top_tiles = [(tiles[keep[j]], float(scores[j])) for j in best]
    return overlay, top_tiles, grid


def main():
    parser = argparse.ArgumentParser(description="Highlight where a term appears in a slide image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--phrase", required=True, help="a name from phrases.yaml (like necrosis), or any text")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--fake", action="store_true", help="use the placeholder scorer (no model needed)")
    parser.add_argument("--checkpoint", default=None, help="CONCH weights path or hf_hub:MahmoodLab/conch")
    parser.add_argument("--out", default=None, help="where to save the heatmap PNG")
    args = parser.parse_args()

    scorer = FakeScorer() if args.fake else ConchScorer(checkpoint_path=args.checkpoint)
    overlay, top_tiles, grid = heatmap_for_image(Image.open(args.image), args.phrase, scorer, args.tile_size)

    safe = re.sub(r"[^a-z0-9]+", "_", args.phrase.lower()).strip("_")
    out = Path(args.out) if args.out else REPO_ROOT / "results" / "figures" / f"heatmap_{safe}_{scorer.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out)

    vals = grid[~np.isnan(grid)]
    print(f"Tiles: {grid.size} total, {len(vals)} tissue, {grid.size - len(vals)} skipped as empty glass")
    print(f"Tiles leaning toward '{args.phrase}' (over 50%): {int((vals >= 0.5).sum())} of {len(vals)}")
    print(f"Top tile scores: {[f'{s:.0%}' for _, s in top_tiles]}")
    print(f"Saved heatmap to {out}")


if __name__ == "__main__":
    main()