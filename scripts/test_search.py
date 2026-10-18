import sys
sys.path.insert(0, ".")
import pandas as pd
from src.search import search

m = pd.read_csv("sample_data/metadata.csv")
row = m[m["caption"].str.contains("carcinoma", case=False)].iloc[0]
img = "sample_data/" + row["image_path"]
print("QUERY:", row["caption"][:100])
for note, w in [("", 0), ("with necrosis", 1.0), ("with dense lymphocytes", 1.0), ("with necrosis", 2.0)]:
    print(f"\n--- note={note!r} weight={w}")
    for r in search(img, note=note, note_weight=w, k=5):
        print(r["score"], "|", r["caption"][:90])