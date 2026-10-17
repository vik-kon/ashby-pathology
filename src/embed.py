import os, sys, numpy as np, torch
from PIL import Image
from tqdm import tqdm
from src.models import load_conch, load_cfg, DEVICE

@torch.no_grad()
def embed_folder(patches_dir, out_dir, batch=64):
    model, preprocess, _ = load_conch()
    paths, labels = [], []
    for cls in sorted(os.listdir(patches_dir)):
        d = os.path.join(patches_dir, cls)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                paths.append(os.path.join(d, f)); labels.append(cls)
    embs = []
    for i in tqdm(range(0, len(paths), batch)):
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i+batch]]).to(DEVICE)
        e = model.encode_image(x, proj_contrast=True, normalize=True)
        embs.append(e.cpu().numpy())
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/img_embs.npy", np.concatenate(embs))
    np.save(f"{out_dir}/paths.npy", np.array(paths))
    np.save(f"{out_dir}/labels.npy", np.array(labels))

if __name__ == "__main__":
    cfg = load_cfg()
    embed_folder(cfg["patches_dir"], cfg["embeddings_dir"])