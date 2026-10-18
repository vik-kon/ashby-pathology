"""Gradio demo for the ashby-pathology case search tool.

Run from the repo root:
    python app/demo.py                  # placeholder results; runs on a laptop with no data or model
    python app/demo.py --real-search    # Vikram's search (needs the index + CONCH, e.g. on Delta)
    python app/demo.py --real-heatmap   # CONCH for slide search
    python app/demo.py --share          # public link (for example when running on Delta)

The search format the app expects (fake_backend.py and Vikram's src/search.py both follow it):
    search(image=None, note="", note_weight=0.3, k=10) -> list of dicts, best first
        each dict: {"image": path or PIL image, "caption": str, "score": float,
                    optional "cancer_type"}
    image only -> search by image
    text only  -> search by description
    both       -> image search, adjusted by the note
"""

import argparse
import math
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.theme import (  # noqa: E402
    CSS, HEAD, THEME, appbar_html, empty_html, footer_html, legend_html,
    rail_heading_html, results_head_html, status_html,
)
from src.heatmap import FakeScorer, draw_overlay, heatmap_for_image, load_phrases  # noqa: E402

EXAMPLE_NOTES = [
    "more aggressive-looking",
    "more inflammation",
    "with necrosis",
    "poorly differentiated",
    "more fibrous stroma",
]


def _missing_model_exit(what, err):
    raise SystemExit(
        f"{what} needs PyTorch and CONCH installed ({err}).\n"
        "On a laptop, run without that flag, or install them: "
        "pip install torch --index-url https://download.pytorch.org/whl/cpu && "
        "pip install git+https://github.com/mahmoodlab/CONCH.git"
    )


def get_search_fn(real):
    if not real:
        from app.fake_backend import search  # becomes the real search if USE_REAL_SEARCH=1 is set

        return search
    try:
        from src.search import search  # Vikram's search: needs CONCH and the embeddings on Delta
    except ImportError as err:
        _missing_model_exit("--real-search", err)
    return search


def get_scorer(real, checkpoint=None):
    if not real:
        return FakeScorer()
    try:
        from src.heatmap import ConchScorer

        return ConchScorer(checkpoint_path=checkpoint)
    except ImportError as err:
        _missing_model_exit("--real-heatmap", err)


def format_tags(value):
    """Specialty tags arrive as a list, as text like "['Pulmonary', 'Breast']", or missing.
    Show them as "Pulmonary, Breast", and show nothing when missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    text = str(value).strip().strip("[]").replace("'", "").replace('"', "")
    return "" if text.lower() in ("", "nan", "none") else text


def build_ui(search_fn, scorer, search_live=False, heatmap_live=False):
    phrase_names = list(load_phrases()["phrases"])
    phrase_choices = [(name.replace("_", " "), name) for name in phrase_names]

    search_status = status_html(
        "Live results from the CONCH index" if search_live else "Sample data, not real cases", search_live
    )
    heatmap_status = status_html(
        "Scored with CONCH" if heatmap_live else "Sample scores, not based on the phrase", heatmap_live
    )

    # ---------- handlers ----------

    def run_search(image, note, note_weight, k):
        note = (note or "").strip()
        if image is None and not note:
            raise gr.Error("Upload a tissue image, type a description, or both.")
        results = search_fn(image, note=note, note_weight=float(note_weight), k=int(k))

        if image is not None and note:
            sub = f"Your image, adjusted by \u201c{note}\u201d. Most similar first; numbers match the table."
        elif image is not None:
            sub = "Cases that look like your image. Most similar first; numbers match the table."
        else:
            sub = f"Cases matching \u201c{note}\u201d. Most similar first; numbers match the table."
        title = f"{len(results)} matches" if results else "No matches"

        gallery = [(r["image"], f"#{i}") for i, r in enumerate(results, 1)]
        table = [
            [i, f"{float(r['score']):.3f}", format_tags(r.get("cancer_type")), r.get("caption", "")]
            for i, r in enumerate(results, 1)
        ]
        return (
            gr.update(value=results_head_html(title, sub, search_status)),
            gr.update(visible=False),
            gr.update(value=gallery, visible=True),
            gr.update(value=table, visible=True),
        )

    def run_heatmap(image, phrase_choice, custom_phrase, tile_size, strength):
        if image is None:
            raise gr.Error("Upload a slide image first.")
        phrase = (custom_phrase or "").strip() or phrase_choice
        if not phrase:
            raise gr.Error("Choose something to look for, or type a phrase.")
        try:
            _, top_tiles, grid = heatmap_for_image(image, phrase, scorer, tile_size=int(tile_size))
        except ValueError as e:
            raise gr.Error(str(e))

        base = image.convert("RGB")
        overlay = draw_overlay(base, grid, int(tile_size), max_alpha=float(strength)) if strength > 0 else base
        vals = grid[~np.isnan(grid)]
        skipped = grid.size - len(vals)
        leaning = int((vals >= 0.5).sum())
        label = phrase.replace("_", " ")
        sub = (
            f"{leaning} of {len(vals)} tissue tiles lean toward \u201c{label}\u201d over the other terms in the list."
            if leaning else
            f"None of the {len(vals)} tissue tiles lean toward \u201c{label}\u201d over the other terms in the list."
        )
        if skipped:
            sub += f" {skipped} empty tiles were skipped."
        head = results_head_html(f"Where \u201c{label}\u201d appears", sub, heatmap_status)
        top = [(tile, f"{score:.0%}") for tile, score in top_tiles]
        state = {"image": base, "grid": grid, "tile": int(tile_size)}
        return (
            gr.update(value=head),
            gr.update(visible=False),
            gr.update(visible=True),
            overlay,
            gr.update(value=legend_html()),
            top,
            state,
        )

    def redraw(strength, state):
        """Fade the overlay without re-scoring. Strength 0 shows the original tissue."""
        if not state:
            return gr.update()
        if strength <= 0:
            return state["image"]
        return draw_overlay(state["image"], state["grid"], state["tile"], max_alpha=float(strength))

    # ---------- layout ----------

    with gr.Blocks(title="Pathology Case Search") as demo:
        gr.HTML(appbar_html())

        with gr.Tabs(elem_classes="main-tabs"):
            # ----- Tab 1: similar cases -----
            with gr.Tab("Similar cases"):
                with gr.Row(equal_height=False, elem_classes="workspace"):
                    with gr.Column(scale=0, min_width=340, elem_classes="rail"):
                        query_img = gr.Image(
                            type="pil", label="Tissue image", height=220, sources=["upload", "clipboard"],
                            placeholder="Drop a tissue image here, or click to upload",
                        )
                        note = gr.Textbox(
                            label="Description or note",
                            placeholder="For example: more aggressive-looking",
                            info="On its own, searches by description. With an image, it adjusts the image search.",
                        )
                        with gr.Row(elem_classes="chips"):
                            for example in EXAMPLE_NOTES:
                                chip = gr.Button(example, size="sm", variant="secondary", min_width=0)
                                chip.click(lambda text=example: text, None, note)
                        gr.HTML('<hr class="rail-divider">' + rail_heading_html("Settings"))
                        note_weight = gr.Slider(0, 1, value=0.3, step=0.05, label="Weight of the note")
                        k = gr.Slider(5, 20, value=10, step=1, label="Number of matches")
                        search_btn = gr.Button("Find similar cases", variant="primary", elem_classes="primary-action")

                    with gr.Column(scale=1, min_width=420):
                        search_head = gr.HTML(results_head_html("Matches", "", search_status))
                        search_empty = gr.HTML(empty_html("search"))
                        gallery = gr.Gallery(
                            show_label=False, columns=5, height="auto", object_fit="cover",
                            visible=False, elem_classes="results-gallery", buttons=["fullscreen"],
                        )
                        table = gr.Dataframe(
                            headers=["Rank", "Score", "Specialty tags", "Caption"],
                            column_widths=["64px", "80px", "170px", "auto"],
                            wrap=True, interactive=False, visible=False, elem_classes="results-table", max_height=760,
                        )
                search_btn.click(
                    run_search, [query_img, note, note_weight, k], [search_head, search_empty, gallery, table]
                )

            # ----- Tab 2: find in a slide -----
            with gr.Tab("Find in a slide (beta)"):
                heat_state = gr.State(None)
                with gr.Row(equal_height=False, elem_classes="workspace"):
                    with gr.Column(scale=0, min_width=340, elem_classes="rail"):
                        slide_img = gr.Image(
                            type="pil", label="Slide image", height=220, sources=["upload", "clipboard"],
                            placeholder="Drop a slide image here, or click to upload",
                        )
                        phrase_choice = gr.Dropdown(
                            choices=phrase_choices,
                            value=phrase_names[0] if phrase_names else None,
                            label="What to look for",
                        )
                        custom_phrase = gr.Textbox(label="Or type your own", placeholder="For example: keratin pearls")
                        gr.HTML('<hr class="rail-divider">' + rail_heading_html("Settings"))
                        tile_size = gr.Slider(128, 512, value=256, step=32, label="Tile size in pixels")
                        heat_btn = gr.Button("Highlight matches", variant="primary", elem_classes="primary-action")

                    with gr.Column(scale=1, min_width=420):
                        heat_head = gr.HTML(results_head_html("Slide view", "", heatmap_status))
                        heat_empty = gr.HTML(empty_html("slide"))
                        with gr.Column(visible=False) as heat_results:
                            overlay = gr.Image(
                                show_label=False, height=520, interactive=False, buttons=["fullscreen", "download"],
                            )
                            with gr.Row(equal_height=True, elem_classes="viewer-tools"):
                                with gr.Column(scale=3, min_width=280):
                                    legend = gr.HTML(legend_html())
                                with gr.Column(scale=2, min_width=240):
                                    strength = gr.Slider(
                                        0, 0.9, value=0.6, step=0.05, label="Overlay strength",
                                        info="Set to 0 to see the original tissue.",
                                    )
                            gr.HTML('<p class="subsection">Best-matching regions</p>')
                            top_gallery = gr.Gallery(
                                show_label=False, columns=5, height="auto", object_fit="cover",
                                elem_classes="results-gallery",
                            )
                heat_btn.click(
                    run_heatmap,
                    [slide_img, phrase_choice, custom_phrase, tile_size, strength],
                    [heat_head, heat_empty, heat_results, overlay, legend, top_gallery, heat_state],
                )
                strength.release(redraw, [strength, heat_state], overlay)

        gr.HTML(footer_html())

    return demo


def launch(demo, share=False, **kwargs):
    """Launch with the app's theme and styles (Gradio 6 takes these in launch)."""
    demo.launch(share=share, theme=THEME, css=CSS, head=HEAD, footer_links=[], **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Pathology case search demo")
    parser.add_argument("--real-search", action="store_true", help="use Vikram's search instead of placeholder results")
    parser.add_argument("--real-heatmap", action="store_true", help="use CONCH instead of the placeholder heatmap scorer")
    parser.add_argument("--checkpoint", default=None, help="CONCH weights path or hf_hub:MahmoodLab/conch")
    parser.add_argument("--share", action="store_true", help="create a public link")
    args = parser.parse_args()

    demo = build_ui(
        get_search_fn(args.real_search),
        get_scorer(args.real_heatmap, args.checkpoint),
        search_live=args.real_search or os.environ.get("USE_REAL_SEARCH") == "1",
        heatmap_live=args.real_heatmap,
    )
    # Let Gradio show result images stored in the shared dataset folders on Delta
    launch(demo, share=args.share, allowed_paths=[
        "/work/nvme/bfqi/shared/datasets",
        "/projects/bfqi/shared/datasets",
    ])


if __name__ == "__main__":
    main()