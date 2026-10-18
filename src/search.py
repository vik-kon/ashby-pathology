from functools import lru_cache
from pathlib import Path
import torch
from PIL import Image
from src.models import load_conch, load_cfg, encode_text, DEVICE
from src.retrieve import Index, fuse

@lru_cache(maxsize=1)
def _load():
    cfg = load_cfg()
    model, pre, tok = load_conch(cfg)
    return model, pre, tok, Index(cfg["embeddings_dir"])

def search(image, note="", note_weight=0.3, k=10, dedup = .97):
    """Same contract as the stand-in. `image` can be a PIL image, a file path, or None."""
    model, pre, tok, index = _load()
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    img_emb = txt_emb = None
    with torch.no_grad():
        if image is not None:
            x = pre(image.convert("RGB")).unsqueeze(0).to(DEVICE)
            img_emb = model.encode_image(x, proj_contrast=True, normalize=True)[0].float().cpu().numpy()
        if note and note.strip():
            txt_emb = encode_text(model, tok, [note])[0].float().cpu().numpy()
    if img_emb is None and txt_emb is None:
        return []
    q = fuse(img_emb, txt_emb, alpha=note_weight)
    
    ids, scores = index.search_ids(q, k * 8)
    keep = []
    for n, gid in enumerate(ids):
        if all(float(index.embs[gid] @ index.embs[ids[m]]) < dedup for m in keep):
            keep.append(n)
        if len(keep) == k:
            break
    out = []
    for n in keep:
        row = index.meta.iloc[ids[n]]
        out.append({"image": row["path"], "caption": row["caption"],
                    "score": round(float(scores[n]), 3), "cancer_type": row["label"]})
    return out