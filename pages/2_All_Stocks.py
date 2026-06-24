from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

import time
import streamlit as st
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# Ticker universe  (S&P-100 sample – extend freely)
# ─────────────────────────────────────────────────────────────────────────────
TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "LLY", "CVX",
    "HD", "MRK", "ABBV", "PEP", "KO", "AVGO", "COST", "MCD", "TMO",
    "CSCO", "ACN", "ABT", "DHR", "NKE", "LIN", "ADBE", "TXN", "NEE",
    "PM", "ORCL", "RTX", "HON", "LOW", "UNP", "QCOM", "IBM", "CAT",
    "AMGN", "INTU", "GS", "BA", "SBUX", "ELV", "MDT", "GILD", "BLK",
    "AXP", "SPGI", "PLD", "ISRG", "ADI", "MDLZ", "BKNG", "REGN", "C",
    "TJX", "CI", "SYK", "ZTS", "VRTX", "MMC", "CB", "AON", "MO", "DUK",
    "SO", "CL", "ITW", "PNC", "USB", "EMR", "FDX", "NSC", "GM", "F",
    "T", "VZ", "TMUS", "CMCSA", "DIS", "NFLX", "CRM", "NOW", "SNOW",
    "PANW", "CRWD", "DDOG", "MDB", "ZS", "OKTA",
]

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="All Stocks — InvestIQ",
    page_icon="📋",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────
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
        --shadow-sm: 0 4px 20px rgba(0,0,0,0.35);
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
        padding-left: clamp(1rem, 3vw, 2.5rem) !important;
        padding-right: clamp(1rem, 3vw, 2.5rem) !important;
    }
    [data-testid="stSidebar"] { display: none !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

    /* Hero */
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
        max-width: 40rem; line-height: 1.65; margin: 0;
    }

    /* Sector headings */
    .sector-title {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.18em;
        text-transform: uppercase; color: var(--accent);
        margin-top: 2rem; margin-bottom: 0.85rem;
        padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
        display: flex; align-items: center; gap: 0.5rem;
    }

    /* Stock cards */
    .card-box {
        background: var(--surface2); backdrop-filter: blur(10px);
        border-radius: 12px; padding: 1.2rem 1rem;
        min-height: 110px; display: flex; align-items: center;
        justify-content: center; text-align: center;
        box-shadow: var(--shadow-sm); border: 1px solid var(--border);
        margin-bottom: 10px; transition: border-color 0.2s, box-shadow 0.2s;
    }
    .card-box:hover {
        border-color: var(--border2);
        box-shadow: var(--shadow-sm), 0 0 16px rgba(0,229,180,0.07);
    }
    .company-name {
        font-family: "Syne", sans-serif; font-size: 0.95rem; font-weight: 700;
        color: var(--ink); line-height: 1.3; margin-bottom: 0.35rem;
    }
    .symbol-text {
        font-family: "IBM Plex Mono", monospace; font-size: 0.78rem;
        color: var(--accent); font-weight: 600; letter-spacing: 0.06em;
    }
    .sector-tag {
        font-size: 0.68rem; color: var(--ink-dim);
        margin-top: 0.25rem; font-style: italic;
    }

    /* Buttons */
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

    /* Inputs */
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
        font-weight: 500 !important; letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
    }
    [data-testid="stAlert"] {
        border-radius: 8px !important; border: 1px solid var(--border) !important;
        background: var(--bg3) !important;
    }
    hr { border: none !important; border-top: 1px solid var(--border) !important; }

    /* Progress / spinner area */
    .stProgress > div > div > div { background-color: var(--accent) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Navbar
# ─────────────────────────────────────────────────────────────────────────────
try:
    from theme import inject_theme, render_navbar
    inject_theme()
    render_navbar(current="browse")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Data fetching  – robust against every known yfinance API quirk
# ─────────────────────────────────────────────────────────────────────────────

def _safe_ticker_info(symbol: str) -> dict:
    """
    Fetch yfinance .info without triggering the 'takes 0 positional arguments'
    error that plagues some yfinance builds.  Falls back gracefully.
    """
    try:
        t = yf.Ticker(symbol)
        # yfinance ≥ 0.2.x exposes .info as a cached property; calling it
        # directly is safe.  Some monkey-patched or older builds wrap it in
        # a no-arg inner function — we handle both.
        raw = t.info
        if callable(raw):          # shouldn't happen, but guard anyway
            raw = raw()
        if isinstance(raw, dict) and raw:
            return raw
    except Exception:
        pass

    # Second attempt: pull just the fast_info (always works)
    try:
        t = yf.Ticker(symbol)
        fi = t.fast_info          # lightweight; never raises on bad tickers
        return {
            "longName":  getattr(fi, "description", None),
            "shortName": symbol,
            "sector":    "Unknown",
            "industry":  "Unknown",
        }
    except Exception:
        pass

    return {}


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_single(symbol: str) -> dict | None:
    """Return a minimal record dict for one ticker, or None on total failure."""
    info = _safe_ticker_info(symbol)
    name = (
        info.get("longName")
        or info.get("shortName")
        or symbol
    )
    sector   = info.get("sector")   or "Unknown"
    industry = info.get("industry") or "Unknown"
    return {
        "symbol":   symbol.upper(),
        "company_name": str(name).strip() or symbol,
        "sector":   str(sector).strip(),
        "industry": str(industry).strip(),
    }


@st.cache_data(ttl=900, show_spinner=False)
def load_all_stocks_cached() -> pd.DataFrame:
    """
    Fetch all tickers in the universe and return a clean DataFrame.
    Uses yf.download-style batch where possible, then per-ticker fallback.
    Results are cached for 15 minutes.
    """
    records: list[dict] = []

    # ── Batch fast_info via yf.Tickers (much faster, single HTTP round-trip)
    try:
        batch = yf.Tickers(" ".join(TICKERS))
        for sym in TICKERS:
            sym_upper = sym.upper()
            try:
                t = batch.tickers.get(sym_upper) or batch.tickers.get(sym)
                if t is None:
                    raise ValueError("not in batch")
                info = t.info  # may still fail for some; caught below
                if callable(info):
                    info = info()
                if not isinstance(info, dict) or not info:
                    raise ValueError("empty info")
                records.append({
                    "symbol":       sym_upper,
                    "company_name": (info.get("longName") or info.get("shortName") or sym_upper).strip(),
                    "sector":       (info.get("sector")   or "Unknown").strip(),
                    "industry":     (info.get("industry") or "Unknown").strip(),
                })
            except Exception:
                # Per-ticker fallback (will hit cache if already fetched)
                r = _fetch_single(sym_upper)
                if r:
                    records.append(r)
            time.sleep(0.05)   # gentle throttle
    except Exception:
        # If even yf.Tickers() fails, go fully sequential
        for sym in TICKERS:
            r = _fetch_single(sym.upper())
            if r:
                records.append(r)
            time.sleep(0.05)

    if not records:
        return pd.DataFrame(columns=["symbol", "company_name", "sector", "industry"])

    df = pd.DataFrame(records).drop_duplicates(subset=["symbol"])
    df["sector"] = df["sector"].replace({"": "Unknown", "None": "Unknown"}).fillna("Unknown")
    return df.sort_values(["sector", "company_name"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="browse-hero">
        <div class="badge">● S&amp;P-100 Universe</div>
        <div class="main-title">Browse stocks</div>
        <div class="sub-title">
            Search by name or symbol, filter by sector, then open any card
            to view charts, financials, and AI-powered Q&amp;A.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load data (with progress bar so the user knows it's working)
# ─────────────────────────────────────────────────────────────────────────────
if "stocks_df" not in st.session_state:
    with st.spinner("Fetching stock data from Yahoo Finance — this takes ~20 s on first load…"):
        st.session_state["stocks_df"] = load_all_stocks_cached()

df: pd.DataFrame = st.session_state["stocks_df"]

# Manual refresh button
if st.button("🔄 Refresh data", type="secondary"):
    load_all_stocks_cached.clear()
    _fetch_single.clear()
    st.session_state.pop("stocks_df", None)
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Search & filter controls
# ─────────────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([2, 1])
with c1:
    search_term = st.text_input(
        "Search by company name or symbol",
        placeholder="e.g. Apple or AAPL",
    ).strip()
with c2:
    sector_options = ["All"] + sorted(
        df["sector"].dropna().unique().tolist()
    ) if not df.empty else ["All"]
    selected_sector = st.selectbox("Filter by sector", sector_options)

# Apply filters
view = df.copy()

if search_term:
    q = search_term.lower()
    view = view[
        view["company_name"].str.lower().str.contains(q, na=False)
        | view["symbol"].str.lower().str.contains(q, na=False)
    ]

if selected_sector != "All":
    view = view[view["sector"] == selected_sector]

# ─────────────────────────────────────────────────────────────────────────────
# Results summary
# ─────────────────────────────────────────────────────────────────────────────
total_loaded = len(df)
st.markdown(
    f"<p style='color:var(--ink-muted);font-size:0.82rem;margin:0.25rem 0 0.5rem;'>"
    f"Showing <strong style='color:var(--accent)'>{len(view)}</strong> of "
    f"<strong style='color:var(--ink)'>{total_loaded}</strong> stocks</p>",
    unsafe_allow_html=True,
)

if view.empty:
    st.warning(
        f'No stocks matched "{search_term or selected_sector}". '
        "Try a different search term or sector."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Stock grid — grouped by sector
# ─────────────────────────────────────────────────────────────────────────────
COLS = 5  # cards per row

for sector, sector_df in view.groupby("sector", sort=True):
    st.markdown(
        f'<div class="sector-title">📂 {sector} '
        f'<span style="color:var(--ink-dim);font-size:0.6rem;">({len(sector_df)})</span></div>',
        unsafe_allow_html=True,
    )

    cards = sector_df.to_dict("records")
    for row_start in range(0, len(cards), COLS):
        row_cards = cards[row_start : row_start + COLS]
        cols = st.columns(COLS)
        for col_idx in range(COLS):
            with cols[col_idx]:
                if col_idx < len(row_cards):
                    company = row_cards[col_idx]
                    sym  = company["symbol"]
                    name = company["company_name"]
                    ind  = company.get("industry", "")

                    st.markdown(
                        f"""
                        <div class="card-box">
                            <div>
                                <div class="company-name">{name}</div>
                                <div class="symbol-text">{sym}</div>
                                {f'<div class="sector-tag">{ind}</div>' if ind and ind != "Unknown" else ""}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(f"Open {sym}", key=f"open_{sym}"):
                        st.session_state["selected_symbol"] = sym
                        st.switch_page("pages/1_Company_Details.py")

st.divider()
st.markdown(
    "<p style='text-align:center;color:var(--ink-dim);font-size:0.75rem;'>"
    "Data sourced from Yahoo Finance via yfinance · Cached 15 min</p>",
    unsafe_allow_html=True,
)