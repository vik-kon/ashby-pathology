import os, numpy as np, pandas as pd, yaml
from src.retrieve import Index

DX = {
    "adenocarcinoma": ["adenocarcinoma"],
    "squamous cell carcinoma": ["squamous cell carcinoma", "squamous carcinoma"],
    "basal cell carcinoma": ["basal cell carcinoma"],
    "melanoma": ["melanoma"],
    "lymphoma": ["lymphoma"],
    "ductal carcinoma": ["ductal carcinoma"],
    "glioma": ["glioma", "glioblastoma", "astrocytoma"],
    "sarcoma": ["sarcoma"],
    "urothelial carcinoma": ["urothelial"],
    "neuroendocrine": ["neuroendocrine", "carcinoid"],
    "hepatocellular carcinoma": ["hepatocellular"],
    "meningioma": ["meningioma"],
}

def dx_label(caption):
    c = str(caption).lower()
    hits = [name for name, kws in DX.items() if any(k in c for k in kws)]
    return hits[0] if len(hits) == 1 else None   # only unambiguous captions get a label

def run(K=10, seed=0):
    cfg = yaml.safe_load(open("configs/paths.yaml"))
    emb_dir = cfg["embeddings_dir"]
    idx = Index(emb_dir, split="train")            # database = train images only
    meta = idx.meta
    meta["dx"] = meta["caption"].map(dx_label)

    q_rows = meta.index[(meta["split"] == "val") & meta["dx"].notna()].to_numpy()
    print(f"queries: {len(q_rows)} val images with a keyword label")
    print(meta.loc[q_rows, "dx"].value_counts().to_string(), "\n")

    _, I = idx.index.search(idx.embs[q_rows], K)    # I indexes into idx.rows
    db_dx = meta["dx"].to_numpy()[idx.rows]
    q_dx = meta["dx"].to_numpy()[q_rows][:, None]
    hits = db_dx[I] == q_dx

    rng = np.random.default_rng(seed)
    rand = db_dx[rng.integers(0, len(db_dx), size=(len(q_rows), K))] == q_dx
    q_labels = meta["dx"].to_numpy()[q_rows]
    print("Per-diagnosis Recall@5 (diagnoses with >=20 queries):")
    macro = []
    for dx in pd.Series(q_labels).value_counts().loc[lambda s: s >= 20].index:
        mask = q_labels == dx
        r, b = hits[mask, :5].any(1).mean(), rand[mask, :5].any(1).mean()
        macro.append((r, b))
        print(f"  {dx:26s} n={mask.sum():4d}  Recall@5 {r:.1%}  random {b:.1%}")
    print(f"Macro-average Recall@5: {np.mean([m[0] for m in macro]):.1%}  (random {np.mean([m[1] for m in macro]):.1%})\n")

    rows = []
    for k in (1, 5, 10):
        r, b = hits[:, :k].any(1).mean(), rand[:, :k].any(1).mean()
        rows.append({"k": k, "recall": round(r, 3), "random_baseline": round(b, 3)})
        print(f"Recall@{k}: {r:.1%}   (random: {b:.1%})")
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/recall.csv", index=False)

if __name__ == "__main__":
    run()