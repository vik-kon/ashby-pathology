import gradio as gr
from PIL import Image
from src.models import load_conch
from src.retrieve import Index

model, _, tok = load_conch()
index = Index()

def search(query, k):
    hits = index.search_text(model, tok, query, int(k))
    return [(Image.open(p), f"{s:.3f}") for p, s in hits]

gr.Interface(
    fn=search,
    inputs=[gr.Textbox(label="Describe the case"), gr.Slider(1, 16, value=8, step=1, label="k")],
    outputs=gr.Gallery(label="Most similar tiles"),
    title="Pathology tile retrieval",
).launch()