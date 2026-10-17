import numpy as np, faiss
from src.models import encode_text

class Index:
    def __init__(self, emb_dir="embeddings"):
        self.embs = np.load(f"{emb_dir}/img_embs.npy").astype("float32")
        self.paths = np.load(f"{emb_dir}/paths.npy")
        self.index = faiss.IndexFlatIP(self.embs.shape[1])   # cosine, since vectors are normalized
        self.index.add(self.embs)

    def search_text(self, model, tokenizer, query, k=8):
        q = encode_text(model, tokenizer, [query]).cpu().numpy().astype("float32")
        scores, idx = self.index.search(q, k)
        return [(self.paths[i], float(s)) for i, s in zip(idx[0], scores[0])]