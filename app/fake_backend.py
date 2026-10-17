"""Stand-in for src/retrieve.py so the app can be built before the real search exists.

It has the SAME function signature and return format the real search should have:

    search(image, note="", note_weight=0.3, k=10) -> list of dicts, best first
        each dict: {"image": file path or PIL image, "caption": str, "score": float,
                    plus optional extras like "cancer_type"}

If sample_data/metadata.csv exists (Vikram's sample files), results come from those
images. Otherwise it returns colored placeholder squares.
"""

import csv
import random
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "sample_data"

# Change these if Vikram's metadata.csv uses different column names
IMAGE_COL = "image_path"
CAPTION_COL = "caption"

FAKE_CAPTIONS = [
    "Invasive ductal carcinoma with solid nests of tumor cells.",
    "Lung adenocarcinoma with acinar gland formation.",
    "Squamous cell carcinoma with keratin pearls.",
    "Tumor with large areas of necrosis.",
    "Dense lymphocytic infiltrate surrounding tumor nests.",
    "Poorly differentiated carcinoma with frequent mitoses.",
    "Fibrous stroma with scattered inflammatory cells.",
    "Normal lung parenchyma with intact alveoli.",
]
FAKE_TYPES = ["BRCA", "LUAD", "LUSC", "COAD"]

_pool = None


def _load_pool():
    meta = SAMPLE_DIR / "metadata.csv"
    rows = []
    if meta.exists():
        with open(meta, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path = Path(row.get(IMAGE_COL, ""))
                if not path.is_absolute():
                    path = SAMPLE_DIR / path
                if path.exists():
                    rows.append({**row, "image": str(path), "caption": row.get(CAPTION_COL, "")})
    if rows:
        return rows

    # No sample data yet: colored placeholder squares with made-up captions
    rng = random.Random(0)
    return [
        {
            "image": Image.new("RGB", (224, 224), (rng.randint(170, 240), rng.randint(80, 170), rng.randint(150, 220))),
            "caption": FAKE_CAPTIONS[i % len(FAKE_CAPTIONS)],
            "cancer_type": FAKE_TYPES[i % len(FAKE_TYPES)],
        }
        for i in range(40)
    ]


def search(image, note="", note_weight=0.3, k=10):
    """Fake search: random scores, with a bump for captions that share a word with the note,
    so the note box and slider visibly do something."""
    global _pool
    if _pool is None:
        _pool = _load_pool()

    rng = random.Random(len(note))
    note_words = {w for w in note.lower().split() if len(w) > 3}
    results = []
    for row in _pool:
        score = rng.uniform(0.2, 0.6)
        if note_words and any(w in row["caption"].lower() for w in note_words):
            score += 0.4 * note_weight
        results.append({**row, "score": round(score, 3)})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]