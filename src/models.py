import os, torch, yaml
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_cfg(path="configs/paths.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class _CompatTokenizer:
    """Lets CONCH's tokenize() work with transformers 5.

    CONCH calls tokenizer.batch_encode_plus(...), which transformers 5 removed.
    Calling the tokenizer directly does the same job, so this wrapper forwards
    batch_encode_plus to it and passes everything else (pad_token_id, ...) through.
    With transformers 4 the wrapper isn't used at all.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def batch_encode_plus(self, texts, **kwargs):
        return self._tokenizer(texts, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def load_conch(cfg=None):
    cfg = cfg or load_cfg()
    token = os.environ.get("HF_TOKEN")  # set in your shell, never in code
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", cfg["conch_checkpoint"], hf_auth_token=token
    )
    model.eval().to(DEVICE)
    tokenizer = get_tokenizer()
    if not hasattr(tokenizer, "batch_encode_plus"):  # transformers 5+
        tokenizer = _CompatTokenizer(tokenizer)
    return model, preprocess, tokenizer

@torch.no_grad()
def encode_text(model, tokenizer, texts):
    tok = tokenize(texts=texts, tokenizer=tokenizer).to(DEVICE)
    emb = model.encode_text(tok)
    return torch.nn.functional.normalize(emb, dim=-1)

# TODO: load_musk() once access is approved (see the xiangjx/musk model card)