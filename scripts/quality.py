"""Drop near-blank / low-tissue images: writes a cleaned copy of the embeddings."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.models import load_cfg

THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25


def tissue_frac(path):
    try:
        im = Image.open(path)
        im.draft("RGB", (160, 160))
        hsv = np.asarray(im.convert("RGB").resize((64, 64)).convert("HSV"), dtype=np.float32) / 255
        return float(((hsv[..., 1] > 0.12) & (hsv[..., 2] < 0.97)).mean())
    except Exception:
        return 0.0


if __name__ == "__main__":
    cfg = load_cfg()
    src = Path(cfg["embeddings_dir"])
    dst = src.parent / (src.name + "_clean")
    dst.mkdir(exist_ok=True)

    meta = pd.read_csv(src / "meta.csv")
    embs = np.load(src / "img_embs.npy")
    assert len(meta) == len(embs)

    with ProcessPoolExecutor(14) as ex:
        frac = np.array(list(ex.map(tissue_frac, meta["path"], chunksize=256)))
    keep = frac >= THRESH

    meta = meta.assign(tissue_frac=frac)
    meta[keep].to_csv(dst / "meta.csv", index=False)
    np.save(dst / "img_embs.npy", embs[keep])
    print(f"kept {keep.sum()} of {len(keep)} ({keep.mean():.1%}) at threshold {THRESH}")
    print("clean dir:", dst)