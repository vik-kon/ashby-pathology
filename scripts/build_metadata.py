import os, sys, pandas as pd, yaml

cfg = yaml.safe_load(open("configs/paths.yaml"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000

# filename -> full path across the 10 image folders
idx = {}
for p in range(1, 11):
    d = f"{cfg['quilt_dir']}/images_part_{p}/quilt_1m"
    for f in os.listdir(d):
        idx[f] = os.path.join(d, f)
print("images found on disk:", len(idx))

cols = ["image_path", "caption", "subset", "split", "pathology", "magnification", "not_histology"]
df = pd.read_csv(cfg["quilt_csv"], usecols=cols)
print("columns ok. split values:\n", df["split"].value_counts(dropna=False))
print("pathology top values:\n", df["pathology"].value_counts(dropna=False).head(15))

bad = df["not_histology"].astype(str).str.lower().isin(["true", "1"])
df = df[~bad]
df["path"] = df["image_path"].map(idx)
df = df.dropna(subset=["path", "caption"])
df["label"] = df["pathology"].fillna("unknown")
df = df.sample(n=min(N, len(df)), random_state=0)[["path", "caption", "label", "subset", "split"]]

os.makedirs("data", exist_ok=True)
df.to_csv(cfg["metadata_csv"], index=False)
print("wrote", len(df), "rows to", cfg["metadata_csv"])