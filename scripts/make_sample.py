import os, shutil, sys, pandas as pd, yaml
cfg = yaml.safe_load(open("configs/paths.yaml"))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
meta = pd.read_csv(f"{cfg['embeddings_dir']}/meta.csv").sample(n, random_state=0)
os.makedirs("sample_data/images", exist_ok=True)
rows = []
for _, r in meta.iterrows():
    name = os.path.basename(r["path"])
    shutil.copy(r["path"], f"sample_data/images/{name}")
    rows.append({"image_path": f"images/{name}", "caption": r["caption"], "cancer_type": r["label"]})
pd.DataFrame(rows).to_csv("sample_data/metadata.csv", index=False)
print("wrote", len(rows), "samples")