import os, argparse, numpy as np, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from src.models import load_conch, load_cfg, DEVICE

class DS(torch.utils.data.Dataset):
    def __init__(self, paths, pre): self.paths, self.pre = paths, pre
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:
            return self.pre(Image.open(self.paths[i]).convert("RGB")), True
        except Exception:
            return torch.zeros(3, 224, 224), False

@torch.no_grad()
def main(limit=None, batch=256, workers=14):
    cfg = load_cfg()
    df = pd.read_csv(cfg["metadata_csv"])
    if limit: df = df.head(limit)
    model, pre, _ = load_conch(cfg)
    dl = torch.utils.data.DataLoader(DS(df["path"].tolist(), pre), batch_size=batch,
                                     num_workers=workers, shuffle=False)
    embs, oks = [], []
    for x, good in tqdm(dl):
        with torch.autocast("cuda", dtype=torch.float16):
            e = model.encode_image(x.to(DEVICE), proj_contrast=True, normalize=True)
        embs.append(e.float().cpu().numpy()); oks.append(good.numpy())
    embs, ok = np.concatenate(embs), np.concatenate(oks)
    out = cfg["embeddings_dir"]; os.makedirs(out, exist_ok=True)
    np.save(f"{out}/img_embs.npy", embs[ok].astype("float16"))
    df[ok].reset_index(drop=True).to_csv(f"{out}/meta.csv", index=False)
    print("embedded", int(ok.sum()), "of", len(df), "images")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)