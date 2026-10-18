"""Visual system for the demo app.

Palette: deep navy for text and primary actions, white and off-white surfaces,
cool greys for secondary surfaces and borders, and teal as the single accent.
Type: Geist throughout, with tabular figures for scores.
"""

import gradio as gr

# ---------- Color scales ----------
NAVY = gr.themes.Color(
    c50="#F2F5FA", c100="#E3E9F2", c200="#C5D0E1", c300="#9DADC8", c400="#6F84A8",
    c500="#4A6089", c600="#33486E", c700="#1C2E52", c800="#14233F", c900="#0D1B33",
    c950="#08111F", name="navy",
)
TEAL = gr.themes.Color(
    c50="#EEF8F9", c100="#E4F2F3", c200="#C2E3E6", c300="#92CCD2", c400="#55AEB7",
    c500="#14838F", c600="#0F7681", c700="#0D6670", c800="#0C535B", c900="#0A434A",
    c950="#062A2F", name="teal",
)
SLATE = gr.themes.Color(
    c50="#F6F8FB", c100="#EEF2F6", c200="#E1E6ED", c300="#C9D2DD", c400="#9AA6B8",
    c500="#6E7B91", c600="#5A6A85", c700="#3E4B62", c800="#2A3446", c900="#1A2130",
    c950="#10151F", name="slate",
)

FONT = [gr.themes.GoogleFont("Geist"), "-apple-system", "BlinkMacSystemFont", "Segoe UI", "system-ui", "sans-serif"]

THEME = gr.themes.Base(
    primary_hue=NAVY,
    secondary_hue=TEAL,
    neutral_hue=SLATE,
    font=FONT,
    font_mono=FONT,
    radius_size=gr.themes.sizes.radius_md,
    text_size=gr.themes.sizes.text_md,
).set(
    # surfaces and text
    body_background_fill="#F6F8FB",
    body_background_fill_dark="#0B1220",
    body_text_color="#0D1B33",
    body_text_color_dark="#E6EBF3",
    body_text_color_subdued="#5A6A85",
    body_text_color_subdued_dark="#9AA6B8",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#111A2B",
    background_fill_secondary="#F6F8FB",
    background_fill_secondary_dark="#0E1626",
    panel_background_fill="#FFFFFF",
    panel_background_fill_dark="#111A2B",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#111A2B",
    block_border_color="#E1E6ED",
    block_border_color_dark="#223049",
    border_color_primary="#E1E6ED",
    border_color_primary_dark="#223049",
    block_shadow="none",
    shadow_drop="0 1px 2px rgba(13, 27, 51, 0.06)",
    block_radius="10px",
    block_label_text_color="#3E4B62",
    block_label_text_color_dark="#C5D0E1",
    block_label_background_fill="#FFFFFF",
    block_label_background_fill_dark="#111A2B",
    block_label_border_color="#E1E6ED",
    block_label_border_color_dark="#223049",
    block_title_text_color="#0D1B33",
    block_title_text_color_dark="#E6EBF3",
    # inputs
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#0E1626",
    input_border_color="#C9D2DD",
    input_border_color_dark="#2A3A57",
    input_border_color_focus="#14838F",
    input_border_color_focus_dark="#55AEB7",
    # buttons
    button_primary_background_fill="#1C2E52",
    button_primary_background_fill_hover="#14233F",
    button_primary_text_color="#FFFFFF",
    button_primary_background_fill_dark="#E6EBF3",
    button_primary_background_fill_hover_dark="#FFFFFF",
    button_primary_text_color_dark="#0D1B33",
    button_primary_shadow="0 1px 2px rgba(13, 27, 51, 0.12)",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F2F5FA",
    button_secondary_text_color="#1C2E52",
    button_secondary_border_color="#C9D2DD",
    button_secondary_background_fill_dark="#111A2B",
    button_secondary_background_fill_hover_dark="#18233A",
    button_secondary_text_color_dark="#E6EBF3",
    button_secondary_border_color_dark="#2A3A57",
    button_large_radius="8px",
    button_small_radius="6px",
    button_transition="background-color 120ms ease, border-color 120ms ease, color 120ms ease",
    # accent
    color_accent="#14838F",
    color_accent_soft="#E4F2F3",
    color_accent_soft_dark="#0F2E33",
    slider_color="#14838F",
    slider_color_dark="#55AEB7",
    loader_color="#14838F",
    loader_color_dark="#55AEB7",
    link_text_color="#0D6670",
    link_text_color_dark="#92CCD2",
    # tables
    table_border_color="#E1E6ED",
    table_border_color_dark="#223049",
    table_even_background_fill="#FFFFFF",
    table_odd_background_fill="#FAFBFD",
    table_even_background_fill_dark="#111A2B",
    table_odd_background_fill_dark="#0F1828",
    table_row_focus="#E4F2F3",
    table_row_focus_dark="#0F2E33",
)

CSS = """
:root {
  --navy: #0D1B33; --navy-700: #1C2E52; --slate: #5A6A85; --line: #E1E6ED; --line-strong: #C9D2DD;
  --surface: #FFFFFF; --canvas: #F6F8FB; --subtle: #EEF2F6;
  --teal: #14838F; --teal-soft: #E4F2F3; --teal-deep: #0D6670;
  --amber: #8A5A12; --amber-soft: #FDF4E3; --amber-line: #F1D9A8;
}
.dark {
  --navy: #E6EBF3; --navy-700: #C5D0E1; --slate: #9AA6B8; --line: #223049; --line-strong: #2A3A57;
  --surface: #111A2B; --canvas: #0B1220; --subtle: #0E1626;
  --teal: #55AEB7; --teal-soft: #0F2E33; --teal-deep: #92CCD2;
  --amber: #F2C77E; --amber-soft: #2A2112; --amber-line: #5A4420;
}

.gradio-container { max-width: 1440px !important; margin: 0 auto !important; padding: 0 24px !important; }
.gradio-container, .gradio-container * { font-feature-settings: "tnum" 1, "cv11" 1; }

/* ---------- App bar ---------- */
.appbar {
  display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  padding: 14px 0 12px; border-bottom: 1px solid var(--line); margin-bottom: 4px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark { width: 28px; height: 28px; flex: none; }
.brand-name { font-size: 1.0625rem; font-weight: 600; color: var(--navy); letter-spacing: -0.01em; }
.brand-sub { font-size: 0.875rem; color: var(--slate); margin-top: 1px; }
.appbar-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.tag {
  display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-size: 0.8125rem; color: var(--slate); background: var(--surface); border: 1px solid var(--line);
}

/* ---------- Tabs ---------- */
.main-tabs > div:first-child { border-bottom: 1px solid var(--line); }
.main-tabs button[role="tab"] {
  font-size: 0.9375rem; font-weight: 500; color: var(--slate); padding: 12px 14px;
  transition: color 120ms ease;
}
.main-tabs button[role="tab"]:hover { color: var(--navy); }
.main-tabs button[role="tab"][aria-selected="true"] { color: var(--navy); font-weight: 600; }

/* ---------- Workspace layout ---------- */
.workspace { gap: 20px !important; margin-top: 16px; align-items: flex-start !important; }

/* Control rail: one surface; the components inside sit flat on it */
.rail {
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 16px !important; gap: 14px !important;
  box-shadow: 0 1px 2px rgba(13, 27, 51, 0.04);
}
.rail .block { border: none !important; box-shadow: none !important; background: transparent !important; padding: 0 !important; }
.rail .form { background: transparent !important; border: 0 !important; box-shadow: none !important; gap: 14px !important; }
.rail .block.image-container, .rail [data-testid="image"] { border: 1px dashed var(--line-strong) !important; border-radius: 10px !important; }
/* Inputs inside the rail keep a clear outline so they read as fields */
.rail textarea, .rail input[type="text"], .rail .wrap-inner {
  border: 1px solid var(--line-strong) !important; border-radius: 8px !important; background: var(--surface) !important;
}
.rail input[type="text"], .rail textarea { padding: 9px 12px !important; }
.rail textarea:focus, .rail input:focus, .rail .wrap-inner:focus-within { border-color: var(--teal) !important; }
.rail-heading { font-size: 0.8125rem; font-weight: 600; color: var(--slate); margin: 4px 0 -4px; }
.rail-divider { border: 0; border-top: 1px solid var(--line); margin: 2px 0; }

.chips { gap: 6px !important; flex-wrap: wrap !important; }
.chips button {
  flex: 0 0 auto !important; border-radius: 999px !important; font-weight: 400 !important;
  font-size: 0.8125rem !important; padding: 4px 10px !important;
  border: 1px solid var(--line-strong) !important;
}
.chips button:hover { border-color: var(--teal) !important; color: var(--teal-deep) !important; background: var(--teal-soft) !important; }
.primary-action { font-size: 0.9375rem !important; font-weight: 600 !important; padding: 11px 0 !important; }

/* ---------- Results ---------- */
.results-head { display: flex; align-items: center; justify-content: space-between; gap: 8px 16px; flex-wrap: wrap; min-height: 32px; }
.results-head > div { flex: 1 1 200px; min-width: 0; }
.results-title { font-size: 1.375rem; line-height: 1.25; font-weight: 600; letter-spacing: -0.015em; color: var(--navy); margin: 0; }
.results-sub { font-size: 0.9375rem; line-height: 1.45; color: var(--slate); margin: 4px 0 0; }
/* Data-source note: plain text, no badge */
.status-note { flex: none; font-size: 0.875rem; color: var(--slate); white-space: nowrap; }
.status-note.live { color: var(--teal-deep); font-weight: 500; }

/* Empty states */
.empty {
  display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
  min-height: 420px; padding: 32px; border: 1px dashed var(--line-strong); border-radius: 12px;
  background: var(--surface); color: var(--slate);
}
.empty svg { width: 36px; height: 36px; color: var(--teal) !important; stroke: var(--teal); margin-bottom: 14px; }
.empty h3 { font-size: 1rem; font-weight: 600; color: var(--navy); margin: 0 0 6px; }
.empty p { font-size: 0.9375rem; line-height: 1.55; max-width: 46ch; margin: 0; }

/* Gallery tiles: size to their content instead of a fixed-height box */
.results-gallery, .results-gallery .gallery-container { height: auto !important; }
.results-gallery .grid-wrap { height: auto !important; min-height: 0 !important; max-height: none !important; overflow: visible !important; }
.results-gallery .gallery-container { overflow: visible !important; }
.results-gallery .thumbnail-item, .results-gallery button.thumbnail-item { border-radius: 8px !important; }
.results-gallery { border-radius: 12px !important; }

/* Table: short columns never break mid-number */
.results-table th:nth-child(-n+2), .results-table td:nth-child(-n+2) { white-space: nowrap; }

/* Heatmap viewer */
.viewer-tools { align-items: center !important; gap: 16px !important; }
.legend { display: flex; align-items: center; gap: 10px; font-size: 0.8125rem; color: var(--slate); padding-top: 10px; }
.legend span { white-space: nowrap; }
.legend .bar {
  flex: 1 1 160px; min-width: 80px; max-width: 260px; height: 8px; border-radius: 999px;
  background: linear-gradient(90deg, #440154, #3B528B, #21918C, #5EC962, #FDE725);
}
.legend b { color: var(--navy); font-weight: 600; }
.legend-note { font-size: 0.8125rem; color: var(--slate); margin: 6px 0 0; }
.result-note p { color: var(--slate); font-size: 0.9375rem; margin: 0; }
.subsection { font-size: 0.9375rem; font-weight: 600; color: var(--navy); margin: 8px 0 0; }

/* Footer */
.foot { color: var(--slate); font-size: 0.8125rem; border-top: 1px solid var(--line); padding: 14px 0 6px; margin-top: 12px; }

/* Focus is always visible for keyboard users */
.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container [role="tab"]:focus-visible {
  outline: 2px solid var(--teal) !important; outline-offset: 2px !important;
}

@media (prefers-reduced-motion: reduce) {
  .gradio-container * { transition: none !important; animation-duration: 0.01ms !important; }
}
@media (max-width: 760px) {
  .gradio-container { padding: 0 12px !important; }
  .rail { min-width: 100% !important; flex: 1 1 100% !important; }
  .empty { min-height: 260px; }
  .results-gallery .grid-container { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
  .results-table table { min-width: 560px; }
}
"""

HEAD = ""  # Geist loads through the theme; nothing extra needed

# ---------- Icons (one line style, 1.6px stroke) ----------
_ICON_SEARCH = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/></svg>'
)
_ICON_LAYERS = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true"><path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/></svg>'
)
_MARK = (
    '<svg class="brand-mark" viewBox="0 0 28 28" aria-hidden="true">'
    '<rect width="28" height="28" rx="7" fill="#1C2E52"/>'
    '<circle cx="12" cy="14" r="6.2" fill="none" stroke="#FFFFFF" stroke-width="1.6"/>'
    '<circle cx="17.5" cy="12" r="2.6" fill="#55AEB7"/></svg>'
)


def appbar_html():
    return (
        '<header class="appbar">'
        f'<div class="brand">{_MARK}<div>'
        '<div class="brand-name">Pathology Case Search</div>'
        '<div class="brand-sub">Find past cases that look like yours</div>'
        "</div></div>"
        '<div class="appbar-meta"><span class="tag">Model: CONCH</span><span class="tag">Research prototype</span></div>'
        "</header>"
    )


def status_html(text, live):
    kind = "live" if live else "sample"
    return f'<span class="status-note {kind}">{text}</span>'


def results_head_html(title, sub, status):
    sub_html = f'<p class="results-sub">{sub}</p>' if sub else ""
    return f'<div class="results-head"><div><h2 class="results-title">{title}</h2>{sub_html}</div>{status}</div>'


def empty_html(kind):
    if kind == "search":
        icon, title = _ICON_SEARCH, "Start with an image, a description, or both"
        body = ("Upload a tissue image or describe what you're looking for. "
                "Matching cases appear here, most similar first.")
    else:
        icon, title = _ICON_LAYERS, "Upload a slide image to search inside it"
        body = ("Choose what to look for, then select Highlight matches. "
                "Tiles that match more strongly are drawn brighter, and you can fade the overlay to see the tissue.")
    return f'<div class="empty">{icon}<h3>{title}</h3><p>{body}</p></div>'


def legend_html():
    """Fixed 0-100% scale: the same color always means the same strength of match."""
    return (
        '<div class="legend"><b>0%</b><span class="bar"></span><b>100%</b></div>'
        '<p class="legend-note">Chance that each tile is this term rather than another term in the list.</p>'
    )


def rail_heading_html(text):
    return f'<p class="rail-heading">{text}</p>'


def footer_html():
    return (
        '<p class="foot">Research prototype for the 2026 Ashby Prize Hackathon at NCSA. '
        "Uses the CONCH pathology model. Not for clinical diagnosis.</p>"
    )