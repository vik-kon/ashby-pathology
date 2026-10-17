"""Feature 2: search inside a slide image for a phrase and draw a heatmap.

Steps: cut the image into tiles -> skip background -> score each tile against
the phrase -> color the tiles by score on top of the image.

Uses the team's code: load_conch() and encode_text() from src/models.py, and the
phrases + templates in prompts/phrases.yaml (templates use {} for the phrase).

Develop on a laptop with the fake scorer (no model or GPU needed):
    python -m src.heatmap --image test_images/he.jpg --phrase necrosis --fake

Use CONCH (on Delta, or on a laptop once your Hugging Face access is approved):
    python -m src.heatmap --image test_images/he.jpg --phrase necrosis
    python -m src.heatmap --image test_images/he.jpg --phrase necrosis --checkpoint hf_hub:MahmoodLab/conch
    (for hf_hub, first run: export HF_TOKEN=your_token)

Note: colors are scaled within each image, so the best-matching tile is always
"hot" even if the phrase is absent. The raw score range is reported alongside the
heatmap so you can tell a real match from a weak one.
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
    """All sentences for one phrase: every synonym dropped into every template.
    A phrase not in phrases.yaml (typed by the user) is used as its own only synonym."""
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


def is_tissue(tile, min_saturation=0.07):
    """Stained tissue is colorful; empty glass is near-white with almost no color saturation."""
    saturation = np.asarray(tile.convert("HSV"))[..., 1] / 255.0
    return float(saturation.mean()) > min_saturation


# ---------- Scorers (same interface: .name and .score(tiles, phrase, cache_key=None)) ----------

class FakeScorer:
    """Placeholder for development only. Ignores the phrase and scores tiles by how dark
    the stain is (dense nuclei look dark), just so the tiling and drawing can be built
    and tested without a model."""

    name = "fake"

    def score(self, tiles, phrase, cache_key=None):
        return np.array([1 - np.asarray(t.convert("L"), dtype=np.float32).mean() / 255 for t in tiles])


class ConchScorer:
    """Real scorer: similarity between each tile's CONCH embedding and the phrase's text embedding."""

    name = "conch"

    def __init__(self, checkpoint_path=None, batch_size=64):
        from src.models import load_conch

        # checkpoint_path=None uses conch_checkpoint from configs/paths.yaml (the Delta path)
        cfg = {"conch_checkpoint": checkpoint_path} if checkpoint_path else None
        self.model, self.preprocess, self.tokenizer = load_conch(cfg)
        self.batch_size = batch_size
        self.cfg = load_phrases()
        self._tile_cache = {}  # embedding the tiles is the slow part, so reuse it across phrases
        self._text_cache = {}

    def _text_vector(self, phrase):
        """Average the embeddings of all sentences for this phrase (a prompt ensemble)."""
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
        if cache_key is not None and cache_key in self._tile_cache:
            tile_emb = self._tile_cache[cache_key]
        else:
            tile_emb = self._embed_tiles(tiles)
            if cache_key is not None:
                self._tile_cache[cache_key] = tile_emb
        return (tile_emb @ self._text_vector(phrase)).float().cpu().numpy()


# ---------- Heatmap ----------

def draw_overlay(img, grid, tile_size, max_alpha=0.6, cmap="viridis"):
    """Color each tissue tile by its score. Higher scores are brighter and more opaque;
    background tiles (NaN) stay transparent. max_alpha=0 returns the tissue unchanged."""
    import matplotlib

    vals = grid[~np.isnan(grid)]
    lo, hi = float(vals.min()), float(vals.max())
    norm = np.nan_to_num((grid - lo) / (hi - lo + 1e-8))

    colors = matplotlib.colormaps[cmap](norm)  # [rows, cols, 4], values 0-1
    colors[..., 3] = np.where(np.isnan(grid), 0.0, max_alpha * (0.08 + 0.92 * norm))
    heat = Image.fromarray((colors * 255).astype(np.uint8))
    heat = heat.resize((grid.shape[1] * tile_size, grid.shape[0] * tile_size), Image.NEAREST)

    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(heat, (0, 0))
    return Image.alpha_composite(base, layer).convert("RGB")


def heatmap_for_image(img, phrase, scorer, tile_size=256, min_saturation=0.07, n_top=5):
    """Run the whole pipeline on one image.

    Returns (overlay_image, top_tiles, grid):
      overlay_image: the input with the heatmap drawn on top
      top_tiles:     [(tile_image, score), ...] for the n_top best-matching regions
      grid:          raw scores per tile position (NaN = background)
    """
    img = img.convert("RGB")
    tiles, coords, (n_rows, n_cols) = tile_image(img, tile_size)
    if not tiles:
        raise ValueError(f"Image is smaller than one {tile_size}px tile. Use a bigger image or a smaller tile size.")

    keep = [i for i, t in enumerate(tiles) if is_tissue(t, min_saturation)]
    if not keep:
        raise ValueError("No tissue found. The image may be mostly background, or try lowering min_saturation.")

    cache_key = f"{hashlib.md5(img.tobytes()).hexdigest()}_{tile_size}_{min_saturation}"
    scores = np.asarray(scorer.score([tiles[i] for i in keep], phrase, cache_key=cache_key), dtype=np.float32)

    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for i, s in zip(keep, scores):
        grid[coords[i]] = s

    overlay = draw_overlay(img, grid, tile_size)
    best = np.argsort(scores)[::-1][:n_top]
    top_tiles = [(tiles[keep[j]], float(scores[j])) for j in best]
    return overlay, top_tiles, grid


def main():
    parser = argparse.ArgumentParser(description="Highlight where a phrase appears in a slide image")
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
    print(f"Tiles: {grid.size} total, {len(vals)} tissue")
    print(f"Raw score range: {vals.min():.3f} to {vals.max():.3f}")
    print(f"Top tile scores: {[round(s, 3) for _, s in top_tiles]}")
    print(f"Saved heatmap to {out}")


if __name__ == "__main__":
    main()