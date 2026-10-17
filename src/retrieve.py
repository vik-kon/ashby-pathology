import numpy as np, pandas as pd, faiss

def _unit(v): return v / np.linalg.norm(v)

def fuse(img_emb=None, txt_emb=None, alpha=0.5):
    """Query vector from an image, a note, or both (alpha = weight of the note)."""
    if img_emb is None: return _unit(txt_emb)
    q = _unit(img_emb)
    if txt_emb is not None: q = q + alpha * _unit(txt_emb)
    return _unit(q)

class Index:
    def __init__(self, emb_dir, split=None):
        self.meta = pd.read_csv(f"{emb_dir}/meta.csv")
        self.embs = np.load(f"{emb_dir}/img_embs.npy").astype("float32")
        mask = np.ones(len(self.meta), bool) if split is None else (self.meta["split"] == split).to_numpy()
        self.rows = np.where(mask)[0]                      # which meta rows are in the index
        self.index = faiss.IndexFlatIP(self.embs.shape[1])  # cosine, vectors are normalized
        self.index.add(self.embs[self.rows])

    def search(self, q, k=8):
        s, i = self.index.search(_unit(q).astype("float32")[None], k)
        return [(self.meta.iloc[self.rows[j]], float(sc)) for j, sc in zip(i[0], s[0])]