import numpy as np, torch, yaml
from src.models import encode_text

def build_class_embeddings(model, tokenizer, yaml_path="prompts/classes.yaml"):
    cfg = yaml.safe_load(open(yaml_path))
    names, embs = [], []
    for cls, synonyms in cfg["classes"].items():
        texts = [t.format(s) for s in synonyms for t in cfg["templates"]]
        e = encode_text(model, tokenizer, texts).mean(0)      # prompt ensemble
        embs.append(torch.nn.functional.normalize(e, dim=-1))
        names.append(cls)
    return names, torch.stack(embs)

def classify(img_embs, class_embs, names, logit_scale=100.0, margin=0.15):
    """img_embs: (N,D) numpy. Returns predictions, with 'uncertain' when top-2 probs are close."""
    sims = torch.from_numpy(img_embs).to(class_embs.device).float() @ class_embs.T
    probs = (sims * logit_scale).softmax(-1).cpu().numpy()
    preds = []
    for p in probs:
        top2 = np.sort(p)[-2:]
        preds.append("uncertain" if top2[1] - top2[0] < margin else names[int(p.argmax())])
    return preds, probs