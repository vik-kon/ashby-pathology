import sys
sys.path.insert(0, ".")
import glob
from src.search import search

img = sorted(glob.glob("sample_data/images/*"))[0]
print("query image:", img)
for note in ["", "more aggressive looking"]:
    print("--- note:", repr(note))
    for r in search(img, note=note, note_weight=0.5, k=5):
        print(r["score"], r["cancer_type"], "|", r["caption"][:100])