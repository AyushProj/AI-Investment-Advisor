import html
import os
from pathlib import Path
import groq

# Load .env from the project folder (Streamlit's cwd is not always the repo root).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
except ImportError:
    pass

import streamlit as st
import pandas as pd

from investiq_data import (
    load_homepage_data,
    load_latest_price_date,
    load_price_history,
    load_price_snapshot,
    load_stock_metrics,
    prepare_merged_universe,
)
from llm_chat import (
    get_gemini_api_key,
    get_gemini_model,
    recommendation_prompt_completion,
)
from price_forecast_ml import arima_forecast_confidence
from theme import inject_theme, render_navbar

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="InvestIQ — AI Investment Advisor",
    page_icon="📈",
    layout="wide"
)

def build_recommendation_prompt(
    merged: pd.DataFrame,
    risk_level: str,
    amount: float,
    horizon_years: int,
    preferred_sector: str,
    wants_dividends: bool,
) -> str:
    sample = merged
    stock_lines = []
    for _, row in sample.iterrows():
        momentum = f"{row['price_momentum']:+.1f}%" if pd.notna(row.get("price_momentum")) else "N/A"
        volatility = f"{row['volatility']:.1f}%" if pd.notna(row.get("volatility")) else "N/A"
        eps = f"{row['tailing_eps']:.2f}" if pd.notna(row.get("tailing_eps")) else "N/A"
        rev_growth = f"{row['revenue_growth']:+.1f}%" if pd.notna(row.get("revenue_growth")) else "N/A"
        dividends = "Yes" if row.get("pays_dividends", 0) == 1 else "No"

        stock_lines.append(
            f"- {row['symbol']} | {row.get('company_name', row['symbol'])} | "
            f"{row['sector']} | {row['industry']} | "
            f"Price: ${row['close']:,.2f} | "
            f"1Y Momentum: {momentum} | "
            f"Volatility: {volatility} | "
            f"Trailing EPS: {eps} | "
            f"Revenue Growth: {rev_growth} | "
            f"Pays Dividends: {dividends}"
        )

    stock_list = "\n".join(stock_lines)
    dividend_pref = "prefers dividend-paying stocks" if wants_dividends else "no dividend preference"

    return f"""You are an AI investment advisor. Based on the investor profile and real stock metrics below, recommend exactly 5 stocks.

Investor Profile:
- Risk tolerance: {risk_level}
- Investment amount: ${amount:,.0f}
- Investment horizon: {horizon_years} year(s)
- Sector preference: {preferred_sector}
- Dividend preference: {dividend_pref}

Metric definitions:
- 1Y Momentum: price change % over last 365 days (positive = growing)
- Volatility: avg daily price range % over last 90 days (higher = riskier)
- Trailing EPS: earnings per share (positive = profitable)
- Revenue Growth: year-over-year revenue change %
- Pays Dividends: whether the stock paid dividends in the last 2 years

Matching guidance:
- Low risk → prefer low volatility, positive EPS, pays dividends
- Medium risk → balanced momentum and volatility, positive EPS
- High risk → higher momentum acceptable, growth-focused
- Short horizon (1-3 yrs) → prefer stable, lower volatility stocks
- Long horizon (7-10 yrs) → growth and momentum more important

Available stocks:
{stock_list}

Instructions:
- Pick exactly 5 stocks that best fit the investor profile using the metrics above.
- Base your reasoning on the actual metric values shown.
- For each pick, provide:
   • a confidence score from 0-100 (how well this fits the profile, given the data),
   • 2-3 key risks specific to this stock or its sector, separated by ';'.
- Reply ONLY in this exact pipe-delimited format, one stock per line, nothing else:
SYMBOL | Reason (mention specific metrics) | CONFIDENCE | risk1; risk2; risk3

Example:
AAPL | Low volatility of 1.2% and positive EPS of 6.43 fit a low-risk 3-year horizon. | 82 | iPhone demand cycles; supply chain in Asia; FX exposure
"""


def parse_recommendation_response(raw: str, merged: pd.DataFrame) -> list[dict]:
    """Parse `SYMBOL | reason | confidence | risks` lines.

    Backward compatible: lines with only `SYMBOL | reason` still work; missing
    fields become None / empty list.
    """
    import re

    results = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-•*0123456789. \t")
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue

        sym = parts[0].upper()
        reason = parts[1] if len(parts) >= 2 else ""
        confidence: int | None = None
        if len(parts) >= 3 and parts[2]:
            m = re.search(r"\d+", parts[2])
            if m:
                try:
                    confidence = max(0, min(100, int(m.group(0))))
                except ValueError:
                    confidence = None
        risks: list[str] = []
        if len(parts) >= 4 and parts[3]:
            risks = [r.strip() for r in re.split(r"[;,]", parts[3]) if r.strip()][:3]

        match = merged[merged["symbol"] == sym]
        if not match.empty:
            row = match.iloc[0]
            results.append({
                "symbol": sym,
                "company_name": row.get("company_name", sym),
                "sector": row["sector"],
                "industry": row["industry"],
                "close": row["close"],
                "reason": reason,
                "confidence": confidence,
                "risks": risks,
            })
    return results


def fallback_recommendations(merged: pd.DataFrame) -> pd.DataFrame:
    top5 = merged.head(5).copy()
    top5["reason"] = "Matches your sector and price preferences."
    return top5[["symbol", "company_name", "sector", "industry", "close", "reason"]].reset_index(drop=True)


def run_llm_recommendations(
    prompt: str,
    merged: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    if not get_gemini_api_key():
        return fallback_recommendations(merged)

    try:
        raw = recommendation_prompt_completion(prompt, model_name)
        results = parse_recommendation_response(raw, merged)
        if not results:
            st.warning(
                "The assistant did not return parseable lines. Showing filtered top picks."
            )
            return fallback_recommendations(merged)
        return pd.DataFrame(results).reset_index(drop=True)
    except Exception as e:
        st.error(f"Assistant error: {e}")
        return fallback_recommendations(merged)


def build_recommendations(
    profile_df: pd.DataFrame,
    latest_prices_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    risk_level: str,
    amount: float,
    horizon_years: int,
    preferred_sector: str,
    wants_dividends: bool,
    prefer_lower_price: bool,
) -> pd.DataFrame:
    """AI-powered stock recommendations."""
    if profile_df.empty or latest_prices_df.empty:
        st.error("No data available to generate recommendations.")
        return pd.DataFrame()

    MAX_STOCKS = 40  # stays within Groq 6000 TPM limit

    if preferred_sector != "Any":
        # -------------------------------------------------------
        # SECTOR SELECTED: pull ONLY that sector's stocks.
        # Never pad with other sectors.
        # -------------------------------------------------------
        merged = prepare_merged_universe(
            profile_df, latest_prices_df, metrics_df,
            preferred_sector, wants_dividends, prefer_lower_price,
        )

        if merged.empty:
            st.warning(f"No stocks found for sector: {preferred_sector}.")
            return pd.DataFrame()

        # Within the sector, still sort by risk-aligned metrics so the
        # best candidates surface at the top before we cap at MAX_STOCKS.
        if "volatility" in merged.columns and merged["volatility"].notna().any():
            if risk_level == "Low":
                merged = merged.sort_values("volatility", ascending=True)
            elif risk_level == "High":
                merged = merged.sort_values("volatility", ascending=False)
            # Medium: keep prepare_merged_universe's default ordering

        merged = merged.head(MAX_STOCKS)

    else:
        # -------------------------------------------------------
        # ANY SECTOR: load full universe, then filter by risk
        # profile using volatility, EPS, and prefer_lower_price.
        # -------------------------------------------------------
        merged = prepare_merged_universe(
            profile_df, latest_prices_df, metrics_df,
            "Any", wants_dividends, prefer_lower_price,
        )

        if merged.empty:
            st.warning("No stocks matched your filters.")
            return pd.DataFrame()

        has_volatility = "volatility" in merged.columns and merged["volatility"].notna().any()
        has_eps = "tailing_eps" in merged.columns and merged["tailing_eps"].notna().any()

        if has_volatility:
            q33 = merged["volatility"].quantile(0.33)
            q66 = merged["volatility"].quantile(0.66)

            if risk_level == "Low":
                filtered = merged[
                    (merged["volatility"] <= q33) &
                    (merged["tailing_eps"] > 0 if has_eps else True)
                ]
            elif risk_level == "Medium":
                filtered = merged[
                    merged["volatility"].between(q33, q66) &
                    (merged["tailing_eps"] > 0 if has_eps else True)
                ]
            else:  # High
                filtered = merged[merged["volatility"] > q66]
        else:
            filtered = merged

        # Fall back to full universe if filter is too aggressive
        if len(filtered) < 5:
            filtered = merged

        # Sort: Low → stable (low vol, positive EPS); High → momentum
        if has_volatility:
            if risk_level == "Low":
                filtered = filtered.sort_values("volatility", ascending=True)
            elif risk_level == "High" and "price_momentum" in filtered.columns:
                filtered = filtered.sort_values("price_momentum", ascending=False)

        merged = filtered.head(MAX_STOCKS)

    resolved_model = get_gemini_model()
    prompt = build_recommendation_prompt(
        merged, risk_level, amount, horizon_years, preferred_sector, wants_dividends,
    )

    if not get_gemini_api_key():
        st.warning(
            "GROQ_API_KEY not set — showing basic filtered suggestions without AI reasoning. "
            "Add one line to a `.env` file next to `app.py`: "
            "`GROQ_API_KEY=your_key` (no spaces around `=`). Or set `GROQ_API_KEY` in "
            "`.streamlit/secrets.toml`. Get a free key at https://console.groq.com/keys — "
            "then restart Streamlit."
        )
        return fallback_recommendations(merged)

    return run_llm_recommendations(prompt, merged, resolved_model)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None
if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = None
if "forecast_open" not in st.session_state:
    st.session_state.forecast_open = {}  # symbol -> bool
if "ml_open" not in st.session_state:
    st.session_state.ml_open = {}  # symbol -> bool  ← NEW

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
try:
    with st.spinner("Loading stock data from yfinance..."):
        df = load_homepage_data()
        latest_prices = load_price_snapshot()
    with st.spinner("Loading stock metrics for smarter recommendations..."):
        stock_metrics = load_stock_metrics()
except Exception as exc:
    st.error(f"Error loading market data: {exc}")
    st.info("Please ensure you have a stable internet connection for yfinance data fetching.")
    st.stop()

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
        --accent3:   #f59e0b;
        --ink:       #f0f4f8;
        --ink-muted: #94a3b8;
        --ink-dim:   #475569;
        --shadow:    0 20px 60px rgba(0, 0, 0, 0.5);
        --shadow-sm: 0 4px 20px rgba(0, 0, 0, 0.35);
        --glow:      0 0 30px rgba(0, 229, 180, 0.12);
    }

    /* ── Base ── */
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

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

    /* ── Hero ── */
    .hero-wrap {
        padding: 2.5rem 0 2rem;
        position: relative;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--accent);
        background: rgba(0, 229, 180, 0.08);
        border: 1px solid rgba(0, 229, 180, 0.25);
        border-radius: 4px;
        padding: 0.3rem 0.75rem;
        margin-bottom: 1.2rem;
    }
    .hero-badge::before {
        content: "●";
        font-size: 0.5rem;
        animation: blink 1.8s ease-in-out infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.2; }
    }
    .main-title {
        font-family: "Syne", sans-serif;
        font-size: clamp(2rem, 5vw, 3.4rem);
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.05;
        color: var(--ink);
        margin: 0 0 0.75rem;
        background: linear-gradient(135deg, #f0f4f8 30%, #00e5b4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-title {
        font-size: clamp(0.9rem, 2vw, 1.05rem);
        color: var(--ink-muted);
        max-width: 38rem;
        line-height: 1.65;
        margin: 0;
    }
    .sub-title strong { color: var(--accent); font-weight: 600; }

    /* ── Nav strip ── */
    .top-nav {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding-top: 2.5rem;
        height: 100%;
    }

    /* ── Buttons ── */
    div.stButton > button[kind="primary"] {
        width: 100% !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.85rem 1.5rem !important;
        font-family: "Syne", sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em !important;
        background: linear-gradient(135deg, #00e5b4 0%, #00c49a 100%) !important;
        color: #080c14 !important;
        box-shadow: 0 0 0 0 rgba(0,229,180,0.4), 0 8px 30px rgba(0,229,180,0.2) !important;
        transition: all 0.25s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 0 3px rgba(0,229,180,0.2), 0 12px 36px rgba(0,229,180,0.3) !important;
        transform: translateY(-1px);
    }
    div.stButton > button[kind="secondary"] {
        border-radius: 7px !important;
        border: 1px solid var(--border2) !important;
        background: rgba(17,24,39,0.6) !important;
        color: var(--ink-muted) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: rgba(0,229,180,0.05) !important;
    }

    /* ── Section card (form wrapper) ── */
    .section-card {
        background: var(--surface);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 2rem 2rem 1.75rem;
        box-shadow: var(--shadow), var(--glow);
        border: 1px solid var(--border);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .section-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        opacity: 0.6;
    }
    .form-section-label {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent);
        margin: 0 0 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .form-section-label::after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--border);
    }
    .form-section-label.spaced { margin-top: 1.5rem; }

    /* ── Form inputs ── */
    .block-container label,
    .stSelectbox label,
    .stSlider label,
    .stCheckbox label,
    .stNumberInput label {
        color: var(--ink-muted) !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
    }
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: var(--bg3) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--ink) !important;
        font-size: 0.95rem !important;
    }
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(0,229,180,0.1) !important;
    }
    .stSlider > div > div > div > div {
        background: var(--accent) !important;
    }
    [data-testid="stSliderThumb"] {
        background: var(--accent) !important;
        border: 2px solid #080c14 !important;
        width: 18px !important; height: 18px !important;
        box-shadow: 0 0 10px rgba(0,229,180,0.5) !important;
    }

    /* ── Checkboxes ── */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] p {
        color: var(--ink-muted) !important;
        font-size: 0.9rem !important;
    }

    /* ── Section heading ── */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 0.5rem 0 1.5rem;
    }
    .section-heading h2 {
        font-family: "Syne", sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0;
        letter-spacing: -0.02em;
    }
    .section-heading .hint {
        font-size: 0.85rem;
        color: var(--ink-dim);
        margin: 0;
        font-family: "IBM Plex Mono", monospace;
    }

    /* ── Suggestion cards ── */
    .suggestion-card {
        background: var(--surface2);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.4rem 1.5rem;
        box-shadow: var(--shadow-sm);
        margin-bottom: 1rem;
        border: 1px solid var(--border);
        transition: border-color 0.25s, box-shadow 0.25s;
        position: relative;
        overflow: hidden;
    }
    .suggestion-card:hover {
        border-color: var(--border2);
        box-shadow: var(--shadow-sm), 0 0 20px rgba(0,229,180,0.06);
    }
    .suggestion-card::after {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, var(--accent), var(--accent2));
        border-radius: 3px 0 0 3px;
    }
    .suggestion-meta {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.85rem;
    }
    .company-name {
        font-family: "Syne", sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--ink);
        line-height: 1.2;
    }
    .symbol-pill {
        display: inline-block;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: var(--accent);
        background: rgba(0,229,180,0.1);
        border: 1px solid rgba(0,229,180,0.25);
        border-radius: 4px;
        padding: 0.18rem 0.5rem;
    }
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.6rem 1rem;
        font-size: 0.83rem;
        color: var(--ink-muted);
        margin-bottom: 0.75rem;
        padding: 0.75rem 1rem;
        background: rgba(0,0,0,0.2);
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    .metric-row strong {
        display: block;
        color: var(--ink-dim);
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.15rem;
    }
    .metric-row span {
        color: var(--ink);
        font-weight: 500;
        font-size: 0.9rem;
    }
    .reason-label {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.4rem;
        opacity: 0.8;
    }
    .reason-box {
        font-size: 0.88rem;
        line-height: 1.6;
        color: var(--ink-muted);
        margin: 0;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        background: rgba(0,229,180,0.04);
        border: 1px solid rgba(0,229,180,0.1);
    }
    .confidence-pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.7rem; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase;
        padding: 0.2rem 0.55rem; border-radius: 4px;
        margin-left: 0.5rem;
    }
    .confidence-high { background: rgba(34,197,94,0.12); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
    .confidence-mid  { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
    .confidence-low  { background: rgba(239,68,68,0.12); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .risks-row {
        display: flex; flex-wrap: wrap; gap: 0.4rem;
        margin-top: 0.6rem;
    }
    .risk-chip {
        font-family: "Inter", sans-serif;
        font-size: 0.72rem; color: var(--ink-muted);
        background: rgba(239,68,68,0.06);
        border: 1px solid rgba(239,68,68,0.18);
        border-radius: 999px;
        padding: 0.18rem 0.65rem;
    }
    .risks-label {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.6rem; font-weight: 600;
        letter-spacing: 0.12em; text-transform: uppercase;
        color: #ef4444; opacity: 0.8;
        margin-top: 0.7rem; margin-bottom: 0.25rem;
    }

    /* ── Inline price forecast (avoids st.metric range / delta coloring bugs) ── */
    .fc-kpi-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.2rem 0 0.35rem;
    }
    .fc-kpi {
        background: rgba(17, 24, 39, 0.72);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.75rem 0.95rem;
        min-width: 0;
    }
    .fc-kpi-lbl {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--ink-dim);
        margin-bottom: 0.3rem;
        line-height: 1.2;
    }
    .fc-kpi-val {
        font-family: "IBM Plex Mono", monospace;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--ink);
        line-height: 1.35;
        word-break: break-word;
    }
    .fc-kpi-val .fc-range-lo,
    .fc-kpi-val .fc-range-mid,
    .fc-kpi-val .fc-range-hi {
        color: var(--ink);
    }
    .fc-kpi-delta {
        font-family: "Inter", system-ui, sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        color: #22c55e;
        margin-top: 0.2rem;
    }
    .fc-kpi-delta.neg { color: #ef4444; }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: var(--bg3) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        margin-bottom: 1.5rem !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--ink-muted) !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-family: "IBM Plex Mono", monospace !important;
        font-size: 1.4rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--ink-dim) !important;
        font-size: 0.75rem !important;
    }

    /* ── Alerts ── */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        border: 1px solid var(--border) !important;
        background: var(--bg3) !important;
    }

    /* ── Divider ── */
    hr {
        margin: 1.5rem 0 !important;
        border: none !important;
        border-top: 1px solid var(--border) !important;
    }

    /* ── Nav pill row ── */
    .nav-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: flex-end;
        align-items: center;
        padding-top: 2.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Apply the active theme palette (after the page CSS so it wins)
inject_theme()
render_navbar(current="home")

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-badge">InvestIQ · AI advisor</div>
        <div class="main-title">Build a portfolio<br>that defines you</div>
        <div class="sub-title">Set your preferences and get AI-powered suggestions. Run a <strong>Portfolio</strong> backtest, plan with the <strong>Goal Planner</strong>, or use <strong>Compare</strong>, <strong>Market IQ</strong>, and <strong>Screener</strong> to dig deeper.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Latest price date — sourced directly from the MAX(report_date) in stock_prices
# so it always reflects the most recent trading day ingested from yfinance.
_latest_date = load_latest_price_date()
st.markdown(f"**Latest data available:** {_latest_date}")

# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<p class="form-section-label">Investor profile</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    risk_level = st.selectbox("Risk tolerance", ["Low", "Medium", "High"])
    amount = st.number_input(
        "Amount to invest (USD)",
        min_value=100,
        value=5000,
        step=100,
    )
    horizon_years = st.slider(
        "Investment horizon (years)",
        min_value=1,
        max_value=10,
        value=3,
    )

with col2:
    sector_options = ["Any"] + sorted(df["sector"].dropna().unique().tolist()) if not df.empty else ["Any"]
    preferred_sector = st.selectbox("Preferred sector", sector_options)
    wants_dividends = st.checkbox("Prefer dividend-oriented names")
    prefer_lower_price = st.checkbox("Prefer lower-priced stocks (vs. median in universe)")

generate = st.button("⚡ Generate AI suggestions", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


def _trend_forecast_fallback(
    hist: pd.DataFrame, horizon_days: int, confidence: float = 0.95
):
    """Simple, always-works forecast: log-linear trend + historical-vol band.

    Used when ARIMA fails to converge. Computes a least-squares trend on
    log(close) over the last ~180 trading days and projects it forward,
    widening the confidence band as sqrt(t) using daily-return std dev.
    """
    import numpy as np
    from pandas.tseries.offsets import BDay
    from dataclasses import dataclass

    @dataclass
    class _Result:
        forecast_dates: list
        point: np.ndarray
        lower: np.ndarray
        upper: np.ndarray
        model_order: tuple
        n_obs: int

    df = hist.sort_values("report_date").dropna(subset=["close"])
    closes = df["close"].astype(float).to_numpy()
    if len(closes) < 30:
        return None
    window = min(180, len(closes))
    y = np.log(closes[-window:])
    x = np.arange(window, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    daily_returns = np.diff(np.log(closes))
    sigma = float(np.std(daily_returns)) if daily_returns.size > 1 else 0.01
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.01

    # Inverse normal at 1 - alpha/2 (95% → 1.96)
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-6 else _z_for_confidence(confidence)

    last_dt = pd.Timestamp(df["report_date"].iloc[-1])
    fdates: list = []
    d = last_dt
    for _ in range(horizon_days):
        d = d + BDay(1)
        fdates.append(d)

    t = np.arange(1, horizon_days + 1, dtype=float)
    log_point = intercept + slope * (window - 1 + t)
    point = np.exp(log_point)
    band = z * sigma * np.sqrt(t)
    lower = np.exp(log_point - band)
    upper = np.exp(log_point + band)

    return _Result(
        forecast_dates=fdates,
        point=point,
        lower=lower,
        upper=upper,
        model_order=("trend",),
        n_obs=int(len(closes)),
    )


def _z_for_confidence(c: float) -> float:
    # Cheap rational approximation; only called for non-default confidences.
    import math
    p = (1.0 + c) / 2.0
    # Beasley-Springer-Moro for the inverse normal CDF
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    q = p - 0.5
    if abs(q) <= 0.425:
        r = q * q
        num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
        den = ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0
        return num / den
    r = math.sqrt(-math.log(min(p, 1 - p)))
    return r if p > 0.5 else -r


def _render_price_forecast(symbol: str, horizon_days: int = 30) -> None:
    """Inline price forecast for one symbol — numbers only (no chart).

    Tries ARIMA first; falls back to a log-linear trend + historical-vol band
    so the demo always renders something useful even if ARIMA cannot converge.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return

    with st.spinner(f"Forecasting {sym}…"):
        hist = load_price_history((sym,), lookback_days=730)

    if hist.empty or len(hist) < 30:
        st.info(
            f"Not enough price history for **{sym}** to run a forecast "
            "(need ~6 weeks minimum)."
        )
        return

    fc = arima_forecast_confidence(hist, horizon=horizon_days, confidence=0.95)
    if fc is None:
        fc = _trend_forecast_fallback(hist, horizon_days, confidence=0.95)
    if fc is None:
        st.info("Could not build a forecast for this series.")
        return

    last_price = float(hist["close"].iloc[-1])
    end_price = float(fc.point[-1])
    change_pct = (end_price / last_price - 1.0) * 100.0 if last_price > 0 else 0.0
    band_low = float(fc.lower[-1])
    band_high = float(fc.upper[-1])

    lo_s = f"${band_low:,.2f}"
    hi_s = f"${band_high:,.2f}"
    delta_cls = "fc-kpi-delta neg" if change_pct < 0 else "fc-kpi-delta"
    st.markdown(
        f"""
<div class="fc-kpi-row">
  <div class="fc-kpi">
    <div class="fc-kpi-lbl">Current</div>
    <div class="fc-kpi-val">${last_price:,.2f}</div>
  </div>
  <div class="fc-kpi">
    <div class="fc-kpi-lbl">+{horizon_days}d forecast</div>
    <div class="fc-kpi-val">${end_price:,.2f}</div>
    <div class="{delta_cls}">{change_pct:+.1f}%</div>
  </div>
  <div class="fc-kpi">
    <div class="fc-kpi-lbl">95% range</div>
    <div class="fc-kpi-val">
      <span class="fc-range-lo">{lo_s}</span><span class="fc-range-mid"> – </span><span class="fc-range-hi">{hi_s}</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ── NEW: Fast cached ML signal for recommendation cards ──────────────────────
# Uses yfinance (same as ML Signal tab) + run_rf_signal (same function).
# risk_score formula is now aligned with homepage definition (avg 20d+60d vol).
# Cached per symbol for 1 hour so repeated renders are instant.
@st.cache_data(ttl=3600)
def get_quick_ml_signal(symbol: str) -> dict | None:
    """
    Run RF classifier for a recommendation card symbol.
    Uses yfinance for price data — same source as ML Signal tab.
    risk_label (Low/Medium/High) is consistent with homepage risk_level.
    """
    try:
        import yfinance as yf
        from price_forecast_ml import run_rf_signal
        hist = yf.Ticker(symbol).history(period="3y", interval="1d")
        if hist.empty or len(hist) < 150:
            return None
        hist = hist.reset_index()
        hist.columns = [c.lower() for c in hist.columns]
        hist = hist.rename(columns={"date": "report_date"})
        hist["report_date"] = pd.to_datetime(hist["report_date"]).dt.tz_localize(None)
        return run_rf_signal(hist)
    except Exception:
        return None


@st.fragment
def render_recommendation_cards(
    recommendations: pd.DataFrame,
    button_key_prefix: str,
    risk_level: str = "Medium",   # ← NEW param: passed from outer scope
) -> None:
    if recommendations.empty:
        st.warning("No recommendations to show.")
        return
    for _, row in recommendations.iterrows():
        sym = html.escape(str(row.get("symbol", "")))
        name = html.escape(str(row.get("company_name", sym)))
        sector = html.escape(str(row.get("sector", "")))
        industry = html.escape(str(row.get("industry", "")))
        reason_raw = row.get("reason") or "Matches your filters and profile."
        reason = html.escape(str(reason_raw))
        try:
            price = float(row["close"])
            price_s = f"${price:,.2f}"
        except (TypeError, ValueError):
            price_s = html.escape(str(row.get("close", "")))

        confidence = row.get("confidence")
        confidence_html = ""
        if confidence is not None and pd.notna(confidence):
            try:
                conf_int = int(confidence)
                tier = (
                    "confidence-high" if conf_int >= 75
                    else "confidence-mid" if conf_int >= 50
                    else "confidence-low"
                )
                confidence_html = (
                    f'<span class="confidence-pill {tier}">'
                    f'AI confidence · {conf_int}%</span>'
                )
            except (TypeError, ValueError):
                confidence_html = ""

        risks = row.get("risks") if isinstance(row.get("risks"), list) else []
        risks_html = ""
        if risks:
            chips = "".join(
                f'<span class="risk-chip">{html.escape(str(r))}</span>'
                for r in risks
            )
            risks_html = (
                '<div class="risks-label">⚠ Key risks</div>'
                f'<div class="risks-row">{chips}</div>'
            )

        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(
                f"""
                <div class="suggestion-card">
                    <div class="suggestion-meta">
                        <span class="company-name">{name}</span>
                        <span class="symbol-pill">{sym}</span>
                        {confidence_html}
                    </div>
                    <div class="metric-row">
                        <div><strong>Sector</strong><span>{sector}</span></div>
                        <div><strong>Industry</strong><span>{industry}</span></div>
                        <div><strong>Last price</strong><span>{price_s}</span></div>
                    </div>
                    <div class="reason-label">↳ Why this pick</div>
                    <p class="reason-box">{reason}</p>
                    {risks_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
            sym_raw = str(row["symbol"])
            details_key = f"open_rec_{button_key_prefix}_{sym_raw}"
            forecast_key = f"forecast_btn_{button_key_prefix}_{sym_raw}"
            ml_key = f"ml_btn_{button_key_prefix}_{sym_raw}"

            if st.button("Details", key=details_key, use_container_width=True):
                st.session_state.selected_symbol = sym_raw
                st.switch_page("pages/1_Company_Details.py")

            is_open = bool(st.session_state.forecast_open.get(sym_raw, False))
            label = "Hide forecast" if is_open else "Price forecast"
            if st.button(label, key=forecast_key, use_container_width=True):
                st.session_state.forecast_open[sym_raw] = not is_open

            # ── NEW: ML Signal button ─────────────────────────────────────
            ml_is_open = bool(st.session_state.ml_open.get(sym_raw, False))
            ml_label = "Hide ML" if ml_is_open else "🤖 ML Signal"
            if st.button(ml_label, key=ml_key, use_container_width=True):
                st.session_state.ml_open[sym_raw] = not ml_is_open

        if st.session_state.forecast_open.get(sym_raw, False):
            _render_price_forecast(sym_raw)

        # ── NEW: ML Signal display ────────────────────────────────────────
        # Uses the same run_rf_signal function as the ML Signal tab.
        # risk_label is now aligned with homepage risk_level definition.
        if st.session_state.ml_open.get(sym_raw, False):
            with st.spinner(f"Running ML analysis for {sym_raw}…"):
                ml = get_quick_ml_signal(sym_raw)

            if ml is None:
                st.caption(f"Not enough price history for ML signal on {sym_raw}.")
            else:
                sig        = ml["signal"]
                risk_label = ml.get("risk_label", "Medium")
                sig_color  = {"Buy":"#00e564","Hold":"#f59e0b","Avoid":"#ef4444"}.get(sig,"#f59e0b")
                sig_emoji  = {"Buy":"📈","Hold":"⏸️","Avoid":"📉"}.get(sig,"⏸️")
                risk_color = {"Low":"#00e564","Medium":"#f59e0b","High":"#ef4444"}.get(risk_label,"#f59e0b")

                # Agreement: compare homepage risk_level vs ML risk_label
                if risk_level == risk_label:
                    agree_text  = "✅ Risk level consistent with your profile"
                    agree_color = "#00e564"
                elif (risk_level == "Low" and risk_label == "High") or \
                     (risk_level == "High" and risk_label == "Low"):
                    agree_text  = "⚠️ Risk mismatch — verify before deciding"
                    agree_color = "#ef4444"
                else:
                    agree_text  = "→ Minor risk difference — review signals"
                    agree_color = "#f59e0b"

                st.markdown(f"""
                <div style="padding:12px 16px;background:rgba(0,0,0,0.2);border-radius:10px;
                            border:1px solid rgba(99,202,183,0.18);margin:6px 0 10px 0;">
                    <div style="font-size:10px;color:#475569;letter-spacing:0.12em;
                                text-transform:uppercase;margin-bottom:8px;">
                        🤖 RF Model — Technical Analysis (RSI · MACD · Momentum)
                    </div>
                    <div style="display:flex;gap:24px;align-items:center;flex-wrap:wrap;">
                        <div>
                            <div style="font-size:10px;color:#94a3b8;margin-bottom:3px;">SIGNAL</div>
                            <div style="font-size:1.1rem;font-weight:800;color:{sig_color};">
                                {sig_emoji} {sig}
                            </div>
                            <div style="font-size:11px;color:#94a3b8;">
                                {ml['confidence']*100:.0f}% confidence
                            </div>
                        </div>
                        <div>
                            <div style="font-size:10px;color:#94a3b8;margin-bottom:3px;">RISK LEVEL</div>
                            <div style="font-size:1.1rem;font-weight:800;color:{risk_color};">
                                {risk_label}
                            </div>
                            <div style="font-size:11px;color:#94a3b8;">{ml['risk_score']}/10</div>
                        </div>
                        <div>
                            <div style="font-size:10px;color:#94a3b8;margin-bottom:3px;">CV ACCURACY</div>
                            <div style="font-size:1rem;font-weight:700;color:#00e5b4;">
                                {ml['cv_accuracy']*100:.1f}%
                            </div>
                            <div style="font-size:11px;color:#94a3b8;">time-series CV</div>
                        </div>
                        <div style="flex:1;text-align:right;min-width:160px;">
                            <div style="font-size:12px;font-weight:600;color:{agree_color};">
                                {agree_text}
                            </div>
                            <div style="font-size:10px;color:#475569;margin-top:3px;">
                                You selected: {risk_level} risk
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ---------------------------------------------------------
# GENERATE RECOMMENDATIONS
# ---------------------------------------------------------
if generate:
    merged = prepare_merged_universe(
        df, latest_prices, stock_metrics,
        preferred_sector, wants_dividends, prefer_lower_price,
    )
    if df.empty or latest_prices.empty:
        st.error("No data available to generate recommendations.")
        st.session_state.recommendation_result = None
    elif merged.empty:
        st.warning("No stocks matched your filters.")
        st.session_state.recommendation_result = None
    else:
        with st.spinner("Asking AI for personalized stock suggestions..."):
            out = build_recommendations(
                profile_df=df,
                latest_prices_df=latest_prices,
                metrics_df=stock_metrics,
                risk_level=risk_level,
                amount=amount,
                horizon_years=horizon_years,
                preferred_sector=preferred_sector,
                wants_dividends=wants_dividends,
                prefer_lower_price=prefer_lower_price,
            )
        st.session_state.recommendation_result = {
            "mode": "single",
            "df": out,
            "risk_level": risk_level,   # ← NEW: store so cards can read it
        }

# ---------------------------------------------------------
# DISPLAY RECOMMENDATIONS
# ---------------------------------------------------------
res = st.session_state.recommendation_result
if res is not None:
    if res["mode"] == "single":
        st.markdown(
            """
            <div class="section-heading">
                <h2>✦ Suggested stocks</h2>
                <span class="hint">AI-ranked · based on your profile</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        rec_df = res["df"]
        if rec_df is None or rec_df.empty:
            st.warning("No recommendations could be generated from the available dataset.")
        else:
            # Pass risk_level so ML signal cards can compare against it
            render_recommendation_cards(
                rec_df,
                "single",
                risk_level=res.get("risk_level", "Medium"),  # ← NEW
            )
