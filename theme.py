"""Dark theme + global background image for InvestIQ.

This module is imported from every page. It re-declares all the CSS custom
properties used across the app on ``:root`` *after* each page's own CSS so
the palette is consistent everywhere, and lays a fixed cityscape image
behind the existing dark gradient.

Usage on every page::

    from theme import inject_theme

    # ... page CSS via st.markdown("<style>...</style>") ...
    inject_theme()
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st


# ── Dark palette (CSS custom properties) ─────────────────────────────────
DARK: dict[str, str] = {
    "--bg":        "#080c14",
    "--bg2":       "#0d1220",
    "--bg3":       "#111827",
    "--surface":   "rgba(17,24,39,0.85)",
    "--surface2":  "rgba(30,41,59,0.7)",
    "--border":    "rgba(99,202,183,0.12)",
    "--border2":   "rgba(99,202,183,0.22)",
    "--accent":    "#00e5b4",
    "--accent2":   "#3b82f6",
    "--accent3":   "#f59e0b",
    "--ink":       "#f0f4f8",
    "--ink-muted": "#94a3b8",
    "--ink-dim":   "#475569",
    "--shadow":    "0 20px 60px rgba(0,0,0,0.5)",
    "--shadow-sm": "0 4px 20px rgba(0,0,0,0.35)",
    "--glow":      "0 0 30px rgba(0,229,180,0.12)",
}


_BG_PATH = Path(__file__).resolve().parent / "assets" / "hero_bg.png"


@lru_cache(maxsize=1)
def _bg_data_uri() -> str:
    """Return the background image as a base64 data URI (cached)."""
    if not _BG_PATH.exists():
        return ""
    data = base64.b64encode(_BG_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def chart_layout() -> dict:
    """Plotly layout dict that matches the dark theme."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f0f4f8",
        xaxis=dict(gridcolor="rgba(99,202,183,0.08)"),
        yaxis=dict(gridcolor="rgba(99,202,183,0.08)"),
    )


def inject_theme() -> None:
    """Apply dark palette + cityscape background on top of any prior CSS."""
    vars_block = "\n".join(
        f"  {k}: {v} !important;" for k, v in DARK.items()
    )
    bg_uri = _bg_data_uri()

    if bg_uri:
        # Image fixed behind everything, with a strong dark overlay so the
        # existing dark UI/text stays readable. The teal/blue radial glows
        # carry over from the original gradient for brand continuity.
        # Even medium dark wash across the whole image (~45%) plus a
        # soft vignette that's a touch darker at the edges. Lets the
        # image read clearly while keeping bold white text legible.
        bg_layers = (
            "radial-gradient(ellipse 120vw 100vh at 50% 50%, "
            "rgba(8,12,20,0.30) 0%, rgba(8,12,20,0.55) 100%),"
            "linear-gradient(180deg, rgba(8,12,20,0.42) 0%, "
            "rgba(8,12,20,0.42) 100%),"
            f"url('{bg_uri}')"
        )
        bg_rules = f"""
        .stApp {{
            background: {bg_layers} !important;
            background-size: auto, auto, cover !important;
            background-position: center, 0 0, center center !important;
            background-attachment: fixed, scroll, fixed !important;
            background-repeat: no-repeat, no-repeat, no-repeat !important;
            color: var(--ink) !important;
        }}
        """
    else:
        # Fallback: original dark gradient if the image is missing.
        bg_rules = """
        .stApp {
            background:
                radial-gradient(ellipse 80vw 60vh at 15% -10%,
                    rgba(0,229,180,0.07) 0%, transparent 60%),
                radial-gradient(ellipse 60vw 50vh at 85% 5%,
                    rgba(59,130,246,0.06) 0%, transparent 55%),
                linear-gradient(180deg,#080c14 0%,#0a0f1c 50%,#080c14 100%)
                !important;
            color: var(--ink) !important;
        }
        """

    st.markdown(
        f"""
        <style>
        :root {{
        {vars_block}
        }}
        {bg_rules}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Unified top navbar ───────────────────────────────────────────────────
# (target path, short label with emoji, unique key token). Labels kept
# short so all 8 items fit on one row on typical desktop viewports.
_NAV_ITEMS: list[tuple[str, str, str]] = [
    ("app.py",                         "🏠 Home",      "home"),
    ("pages/2_All_Stocks.py",          "📊 Browse",    "browse"),
    ("pages/3_Compare_Stocks.py",      "⚖️ Compare",   "compare"),
    ("pages/5_Stock_Screener.py",      "🔬 Screener",  "screener"),
    ("pages/4_Market_Intelligence.py", "🧠 Market IQ", "marketiq"),
    ("pages/6_News_Sentiment.py",      "📰 News",      "news"),
    ("pages/7_Portfolio_Simulator.py", "💼 Portfolio", "portfolio"),
    ("pages/8_Goal_Planner.py",        "🎯 Goals",     "goals"),
]


_NAV_CSS = """
<style>
/* Navbar buttons (type=secondary, keys prefixed nav_) get compact padding
   and disabled = highlighted "current page" treatment. We scope to
   disabled secondary buttons so the overall app isn't affected. */
div.stButton > button[kind="secondary"]:disabled {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(0,229,180,0.12) !important;
    opacity: 1 !important;
    cursor: default !important;
}
</style>
"""


def render_navbar(current: str | None = None) -> None:
    """Render the shared top navbar.

    Pass ``current`` as the third-element token from ``_NAV_ITEMS`` to
    highlight + disable the button for the current page.
    """
    st.markdown(_NAV_CSS, unsafe_allow_html=True)
    cols = st.columns(len(_NAV_ITEMS))
    for col, (target, label, token) in zip(cols, _NAV_ITEMS):
        with col:
            is_current = token == current
            if st.button(
                label,
                key=f"nav_{token}",
                type="secondary",
                use_container_width=True,
                disabled=is_current,
            ):
                st.switch_page(target)
    st.markdown(
        "<div style='margin:0.25rem 0 0.75rem 0;'></div>",
        unsafe_allow_html=True,
    )
