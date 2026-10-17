import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from src.models import load_conch
from src.zeroshot import build_class_embeddings, classify

def run(emb_dir="embeddings", margin=0.15):
    model, _, tok = load_conch()
    names, class_embs = build_class_embeddings(model, tok)
    X = np.load(f"{emb_dir}/img_embs.npy"); y = np.load(f"{emb_dir}/labels.npy")
    preds, _ = classify(X, class_embs, names, margin=margin)
    preds = np.array(preds); answered = preds != "uncertain"
    print(f"coverage: {answered.mean():.2%}")
    print(f"accuracy (all, uncertain counts wrong): {accuracy_score(y, preds):.2%}")
    print(f"accuracy (answered only): {accuracy_score(y[answered], preds[answered]):.2%}")
    print(confusion_matrix(y[answered], preds[answered], labels=names))

if __name__ == "__main__":
    run()