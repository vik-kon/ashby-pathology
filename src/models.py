import os, torch, yaml
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_cfg(path="configs/paths.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_conch(cfg=None):
    cfg = cfg or load_cfg()
    token = os.environ.get("HF_TOKEN")  # set in your shell, never in code
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", cfg["conch_checkpoint"], hf_auth_token=token
    )
    model.eval().to(DEVICE)
    return model, preprocess, get_tokenizer()

@torch.no_grad()
def encode_text(model, tokenizer, texts):
    tok = tokenize(texts=texts, tokenizer=tokenizer).to(DEVICE)
    emb = model.encode_text(tok)
    return torch.nn.functional.normalize(emb, dim=-1)

# TODO: load_musk() once access is approved (see the xiangjx/musk model card)