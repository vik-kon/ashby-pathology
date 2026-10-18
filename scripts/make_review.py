import os, sys, shutil, html
sys.path.insert(0, ".")
import numpy as np, yaml
from src.retrieve import Index
from src.eval import dx_label

dx_name = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
cfg = yaml.safe_load(open("configs/paths.yaml"))
idx = Index(cfg["embeddings_dir"], split="train")
meta = idx.meta
meta["dx"] = meta["caption"].map(dx_label)

q_rows = meta.index[(meta["split"] == "val") & (meta["dx"] == dx_name)].to_numpy()
_, I = idx.index.search(idx.embs[q_rows], 5)
db_dx = meta["dx"].to_numpy()[idx.rows]
failed = q_rows[~(db_dx[I] == dx_name).any(1)]            # queries with no same-diagnosis hit in top 5
rng = np.random.default_rng(0)
pick = rng.choice(len(failed), size=min(n, len(failed)), replace=False)

out = f"review/{dx_name.replace(' ', '_')}"
os.makedirs(f"{out}/img", exist_ok=True)

def cell(path, tag, caption, extra=""):
    name = f"{tag}{os.path.splitext(path)[1]}"
    shutil.copy(path, f"{out}/img/{name}")
    return f"<td><img src='img/{name}' width='210'><br><small>{html.escape(str(caption)[:220])}</small><br><b>{extra}</b></td>"

rows_html = []
for r, p in enumerate(pick, 1):
    q = failed[p]
    qi = list(q_rows).index(q)
    cells = [cell(meta.loc[q, "path"], f"q{r}", meta.loc[q, "caption"], f"QUERY #{r} ({dx_name})")]
    for j, gid in enumerate(idx.rows[I[qi]]):
        lab = meta.loc[gid, "dx"] or "(no keyword label)"
        cells.append(cell(meta.loc[gid, "path"], f"q{r}_n{j+1}", meta.loc[gid, "caption"], f"neighbor {j+1}: {lab}"))
    rows_html.append("<tr>" + "".join(cells) + "</tr>")

open(f"{out}/index.html", "w").write(
    f"<html><body><h2>{dx_name}: queries where the top 5 had NO {dx_name}</h2>"
    f"<table border='1' cellpadding='4' valign='top'>{''.join(rows_html)}</table></body></html>")
print(f"wrote {len(pick)} of {len(failed)} failed queries to {out}/index.html")