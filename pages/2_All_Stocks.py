from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

import streamlit as st
import pandas as pd
from investiq_data import load_homepage_data

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="All Stocks",
    page_icon="📋",
    layout="wide"
)

# ---------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------
@st.cache_data
def load_all_stocks() -> pd.DataFrame:
    """Load all stocks from yfinance via investiq_data."""
    return load_homepage_data()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "show_search_popup" not in st.session_state:
    st.session_state.show_search_popup = False
if "last_invalid_search" not in st.session_state:
    st.session_state.last_invalid_search = ""

# ---------------------------------------------------------
# PAGE STYLES
# ---------------------------------------------------------
st.markdown("""
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
        --shadow-sm: 0 4px 20px rgba(0, 0, 0, 0.35);
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
        padding-top: 1rem !important;
        padding-bottom: 4rem !important;
        padding-left: clamp(1rem, 3vw, 2.5rem) !important;
        padding-right: clamp(1rem, 3vw, 2.5rem) !important;
    }
    [data-testid="stSidebar"] { display: none !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

    /* ── Hero ── */
    .browse-hero { padding: 1.5rem 0 1rem; }
    .browse-hero .badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em;
        text-transform: uppercase; color: var(--accent);
        background: rgba(0,229,180,0.08); border: 1px solid rgba(0,229,180,0.25);
        border-radius: 4px; padding: 0.3rem 0.75rem; margin-bottom: 0.75rem;
    }
    .main-title {
        font-family: "Syne", sans-serif;
        font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 800;
        letter-spacing: -0.04em; line-height: 1.05; margin: 0 0 0.6rem;
        background: linear-gradient(135deg, #f0f4f8 30%, #00e5b4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .sub-title {
        font-size: clamp(0.9rem, 2vw, 1rem); color: var(--ink-muted);
        max-width: 36rem; line-height: 1.65; margin: 0;
    }

    /* ── Sector headings ── */
    .sector-title {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.18em;
        text-transform: uppercase; color: var(--accent);
        margin-top: 2rem; margin-bottom: 0.85rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
        display: flex; align-items: center; gap: 0.5rem;
    }
    .sector-title::after { content: ""; flex: 1; height: 1px; background: var(--border); }

    /* ── Stock cards ── */
    .card-box {
        background: var(--surface2); backdrop-filter: blur(10px);
        border-radius: 12px; padding: 1.2rem 1rem;
        min-height: 110px; display: flex; align-items: center;
        justify-content: center; text-align: center;
        box-shadow: var(--shadow-sm); border: 1px solid var(--border);
        margin-bottom: 10px; transition: border-color 0.2s, box-shadow 0.2s;
    }
    .card-box:hover { border-color: var(--border2); box-shadow: var(--shadow-sm), 0 0 16px rgba(0,229,180,0.07); }
    .company-name {
        font-family: "Syne", sans-serif; font-size: 0.95rem; font-weight: 700;
        color: var(--ink); line-height: 1.3; margin-bottom: 0.35rem;
    }
    .symbol-text {
        font-family: "IBM Plex Mono", monospace; font-size: 0.78rem;
        color: var(--accent); font-weight: 600; letter-spacing: 0.06em;
    }

    /* ── Buttons ── */
    div.stButton > button {
        width: 100%; border-radius: 7px !important;
        border: 1px solid var(--border2) !important;
        background: rgba(17,24,39,0.6) !important; color: var(--ink-muted) !important;
        font-weight: 500 !important; font-size: 0.82rem !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        border-color: var(--accent) !important; color: var(--accent) !important;
        background: rgba(0,229,180,0.05) !important;
    }
    div.stButton > button[kind="secondary"] {
        background: rgba(17,24,39,0.6) !important;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input {
        background: var(--bg3) !important; border: 1px solid var(--border) !important;
        border-radius: 8px !important; color: var(--ink) !important;
    }
    .stTextInput > div > div > input:focus { border-color: var(--accent) !important; }
    .stSelectbox > div > div {
        background: var(--bg3) !important; border: 1px solid var(--border) !important;
        border-radius: 8px !important; color: var(--ink) !important;
    }
    .block-container label, .stSelectbox label, .stTextInput label {
        color: var(--ink-muted) !important; font-size: 0.8rem !important;
        font-weight: 500 !important; letter-spacing: 0.03em !important; text-transform: uppercase !important;
    }
    [data-testid="stAlert"] {
        border-radius: 8px !important; border: 1px solid var(--border) !important; background: var(--bg3) !important;
    }
    hr { border: none !important; border-top: 1px solid var(--border) !important; }
</style>
""", unsafe_allow_html=True)

from theme import inject_theme, render_navbar  # noqa: E402
inject_theme()
render_navbar(current="browse")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = load_all_stocks()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="browse-hero">
        <div class="badge">● Universe</div>
        <div class="main-title">Browse stocks</div>
        <div class="sub-title">Search and filter by sector, then open a ticker for charts and AI Q&amp;A.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SEARCH & FILTER
# ---------------------------------------------------------
c1, c2 = st.columns([2, 1])
with c1:
    search_term = st.text_input("Search by company name, symbol, or sector").strip()
with c2:
    sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist()) if not df.empty else ["All"]
    selected_sector = st.selectbox("Filter by Sector", sectors)

working_df = df.copy()

if search_term:
    q = search_term.lower()
    working_df = working_df[
        working_df["company_name"].str.lower().str.contains(q, na=False) |
        working_df["symbol"].str.lower().str.contains(q, na=False) |
        working_df["sector"].str.lower().str.contains(q, na=False)
    ]

if selected_sector != "All":
    working_df = working_df[working_df["sector"] == selected_sector]

if search_term and working_df.empty:
    st.session_state.show_search_popup = True
    st.session_state.last_invalid_search = search_term

if st.session_state.show_search_popup:
    st.error(f'The company "{st.session_state.last_invalid_search}" does not exist in the current dataset.')
    if st.button("Close Search Message"):
        st.session_state.show_search_popup = False

# ---------------------------------------------------------
# STOCK GRID
# ---------------------------------------------------------
if not working_df.empty:
    grouped = working_df.groupby("sector", sort=True)

    for sector, sector_df in grouped:
        st.markdown(f'<div class="sector-title">{sector}</div>', unsafe_allow_html=True)
        cards = sector_df.to_dict("records")

        for i in range(0, len(cards), 5):
            row_cards = cards[i:i + 5]
            cols = st.columns(5)

            for j in range(5):
                with cols[j]:
                    if j < len(row_cards):
                        company = row_cards[j]
                        st.markdown(
                            f"""
                            <div class="card-box">
                                <div>
                                    <div class="company-name">{company['company_name']}</div>
                                    <div class="symbol-text">{company['symbol']}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(f"Open {company['symbol']}", key=f"open_{company['symbol']}"):
                            st.session_state.selected_symbol = company["symbol"]
                            st.switch_page("pages/1_Company_Details.py")
else:
    st.warning("No companies found.")