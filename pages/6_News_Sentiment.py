"""Near–real-time Yahoo Finance news with VADER sentiment scores."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from news_sentiment import fetch_yahoo_news_with_sentiment

st.set_page_config(
    page_title="News & sentiment — InvestIQ",
    page_icon="📰",
    layout="wide",
)

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg:        #080c14;
        --bg2:       #0d1220;
        --bg3:       #111827;
        --surface:   rgba(17, 24, 39, 0.85);
        --surface2:  rgba(30, 41, 59, 0.7);
        --border:    rgba(99, 202, 183, 0.12);
        --border2:   rgba(99, 202, 183, 0.22);
        --accent:    #00e5b4;
        --accent2:   #3b82f6;
        --ink:       #f0f4f8;
        --ink-muted: #94a3b8;
        --ink-dim:   #475569;
        --shadow:    0 20px 60px rgba(0, 0, 0, 0.5);
        --glow:      0 0 30px rgba(0, 229, 180, 0.12);
    }
    .stApp {
        font-family: "Inter", system-ui, sans-serif !important;
        background:
            radial-gradient(ellipse 80vw 60vh at 15% -10%, rgba(0,229,180,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 60vw 50vh at 85% 5%,  rgba(59,130,246,0.06) 0%, transparent 55%),
            linear-gradient(180deg, #080c14 0%, #0a0f1c 50%, #080c14 100%) !important;
        color: var(--ink) !important;
    }
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 80rem !important;
        padding-top: 1rem !important; padding-bottom: 4rem !important;
        padding-left: clamp(1rem, 3vw, 2.5rem) !important; padding-right: clamp(1rem, 3vw, 2.5rem) !important;
    }
    [data-testid="stSidebar"] { display: none !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

    /* ── Hero ── */
    .news-hero { padding: 1.5rem 0 0.75rem; }
    .news-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em;
        text-transform: uppercase; color: var(--accent);
        background: rgba(0,229,180,0.08); border: 1px solid rgba(0,229,180,0.25);
        border-radius: 4px; padding: 0.3rem 0.75rem; margin-bottom: 0.75rem;
    }
    .news-title {
        font-family: "Syne", sans-serif;
        font-size: clamp(2rem, 5vw, 3rem); font-weight: 800;
        letter-spacing: -0.04em; margin: 0 0 0.5rem;
        background: linear-gradient(135deg, #f0f4f8 30%, #00e5b4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .news-sub { font-size: 0.9rem; color: var(--ink-muted); max-width: 44rem; line-height: 1.6; }

    /* ── Buttons ── */
    div.stButton > button[kind="primary"] {
        border-radius: 8px !important; border: none !important;
        background: linear-gradient(135deg, #00e5b4 0%, #00c49a 100%) !important;
        color: #080c14 !important; font-weight: 700 !important; font-family: "Syne", sans-serif !important;
        box-shadow: 0 8px 30px rgba(0,229,180,0.2) !important;
    }
    div.stButton > button[kind="secondary"] {
        border-radius: 7px !important; border: 1px solid var(--border2) !important;
        background: rgba(17,24,39,0.6) !important; color: var(--ink-muted) !important;
        font-weight: 500 !important; font-size: 0.85rem !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: var(--accent) !important; color: var(--accent) !important;
        background: rgba(0,229,180,0.05) !important;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input {
        background: var(--bg3) !important; border: 1px solid var(--border) !important;
        border-radius: 8px !important; color: var(--ink) !important;
    }
    .stTextInput > div > div > input:focus { border-color: var(--accent) !important; }
    .stSlider > div > div > div > div { background: var(--accent) !important; }
    [data-testid="stSliderThumb"] {
        background: var(--accent) !important; border: 2px solid #080c14 !important;
        box-shadow: 0 0 10px rgba(0,229,180,0.5) !important;
    }
    .block-container label, .stTextInput label, .stSlider label {
        color: var(--ink-muted) !important; font-size: 0.8rem !important;
        font-weight: 500 !important; letter-spacing: 0.03em !important; text-transform: uppercase !important;
    }

    /* ── Toggle ── */
    .stToggle > label > div[data-testid="stMarkdownContainer"] p { color: var(--ink-muted) !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px !important; border: 1px solid var(--border) !important; }

    /* ── Alerts / divider ── */
    [data-testid="stAlert"] {
        border-radius: 8px !important; border: 1px solid var(--border) !important; background: var(--bg3) !important;
    }
    hr { border: none !important; border-top: 1px solid var(--border) !important; }
</style>
""",
    unsafe_allow_html=True,
)

from theme import inject_theme, render_navbar  # noqa: E402
inject_theme()
render_navbar(current="news")

st.markdown(
    """
    <div class="news-hero">
        <div class="news-badge">● Yahoo Finance + VADER</div>
        <div class="news-title">News sentiment</div>
        <div class="news-sub">
            Track recent headlines for any ticker and score each article using
            VADER compound sentiment (−1 to +1).
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

sym = st.text_input("Ticker symbol", value="AAPL", max_chars=8).strip().upper()
n_headlines = st.slider("Max headlines", min_value=5, max_value=40, value=20)
auto = st.toggle("Auto-refresh every 2 minutes", value=False)

if auto:

    @st.fragment(run_every=timedelta(minutes=2))
    def _news_fragment() -> None:
        try:
            df = fetch_yahoo_news_with_sentiment(sym, limit=n_headlines)
        except Exception as exc:
            st.error(f"Could not load news: {exc}")
            return
        st.caption(f"Updated {datetime.now().strftime('%H:%M:%S')} · {sym}")
        if df.empty:
            st.info("No news returned for this symbol (try another ticker).")
            return
        st.dataframe(
            df,
            column_config={
                "sentiment": st.column_config.NumberColumn(
                    "Sentiment",
                    help="VADER compound score (−1 to +1)",
                    format="%.3f",
                ),
                "url": st.column_config.LinkColumn("Link"),
            },
            use_container_width=True,
            height=min(560, 48 * (len(df) + 2)),
            hide_index=True,
        )

    _news_fragment()
else:
    go = st.button("Load news", type="primary")
    if go:
        try:
            df = fetch_yahoo_news_with_sentiment(sym, limit=n_headlines)
        except Exception as exc:
            st.error(f"Could not load news: {exc}")
        else:
            if df.empty:
                st.info("No news returned for this symbol.")
            else:
                st.dataframe(
                    df,
                    column_config={
                        "sentiment": st.column_config.NumberColumn(
                            "Sentiment",
                            help="VADER compound score (−1 to +1)",
                            format="%.3f",
                        ),
                        "url": st.column_config.LinkColumn("Link"),
                    },
                    use_container_width=True,
                    height=min(560, 48 * (len(df) + 2)),
                    hide_index=True,
                )

st.divider()