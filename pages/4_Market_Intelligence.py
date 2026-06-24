"""Market-wide Databricks SQL views: sectors, movers, volume, filings."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Display helpers: plain-language column labels + consistent table chrome
# ---------------------------------------------------------------------------

_MI_DF_KWARGS: dict = {
    "hide_index": True,
    "use_container_width": True,
}


def _rename_movers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mapping = {
        "company_name": "Company",
        "symbol": "Ticker",
        "last_close": "Last price",
        "price_momentum": "1-year change (%)",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def _movers_column_config() -> dict:
    return {
        "Company": st.column_config.TextColumn("Company", width="large"),
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Last price": st.column_config.NumberColumn(
            "Last price",
            format="$%.2f",
            help="Latest closing price in the warehouse.",
        ),
        "1-year change (%)": st.column_config.NumberColumn(
            "1-year change (%)",
            format="%.1f",
            help="Approximate one-year price change (latest vs anchor date about a year back).",
        ),
    }


def _rename_volume(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mapping = {
        "symbol": "Ticker",
        "last_day_volume": "Latest volume",
        "avg_30d_volume": "Recent average",
        "volume_vs_30d_avg": "Compared to average",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def _volume_column_config() -> dict:
    return {
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Latest volume": st.column_config.NumberColumn(
            "Latest volume",
            format="%.0f",
            help="Shares traded on the most recent day in the data.",
        ),
        "Recent average": st.column_config.NumberColumn(
            "Recent average",
            format="%.0f",
            help="Mean volume over the prior 30 trading sessions.",
        ),
        "Compared to average": st.column_config.NumberColumn(
            "Compared to average (times)",
            format="%.2f",
            help="Latest volume divided by that average (above 1 = busier than usual).",
        ),
    }


def _rename_filings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mapping = {
        "symbol": "Ticker",
        "company_name": "Company",
        "filing_date": "Filing date",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def _filings_column_config() -> dict:
    return {
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Company": st.column_config.TextColumn("Company", width="large"),
        "Filing date": st.column_config.DatetimeColumn(
            "Filing date",
            format="YYYY-MM-DD",
            step=86400,
        ),
    }


def _rename_dividends(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mapping = {
        "symbol": "Ticker",
        "report_date": "Paid on",
        "amount": "Per share",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def _dividends_column_config() -> dict:
    return {
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Paid on": st.column_config.DatetimeColumn(
            "Paid on",
            format="YYYY-MM-DD",
            step=86400,
        ),
        "Per share": st.column_config.NumberColumn(
            "Per share",
            format="$%.4f",
            help="Cash dividend amount per share.",
        ),
    }


def _rename_revenue(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mapping = {
        "symbol": "Ticker",
        "item_value": "Revenue",
        "report_date": "Period end",
    }
    out = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    keep = [c for c in ("Ticker", "Revenue", "Period end") if c in out.columns]
    return out[keep]


def _revenue_column_config() -> dict:
    return {
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Revenue": st.column_config.NumberColumn(
            "Revenue",
            format="%.0f",
            help="Reported revenue for the latest annual row we matched.",
        ),
        "Period end": st.column_config.DatetimeColumn(
            "Period end",
            format="YYYY-MM-DD",
            step=86400,
        ),
    }


def _sectors_column_config() -> dict:
    return {
        "Sector": st.column_config.TextColumn("Sector", width="medium"),
        "Stocks": st.column_config.NumberColumn(
            "Stocks",
            format="%d",
            help="How many tickers in this sector in our universe.",
        ),
        "Typical 1-year return (%)": st.column_config.NumberColumn(
            "Typical 1-year return (%)",
            format="%.1f",
            help="Average of each stock's one-year price change.",
        ),
        "Typical daily swing (%)": st.column_config.NumberColumn(
            "Typical daily swing (%)",
            format="%.1f",
            help="Average day-to-day high-low range vs price (last ~90 days).",
        ),
    }

from investiq_data import (
    load_recent_dividend_events,
    load_recent_sec_filings,
    load_sector_benchmarks,
    load_statement_highlights,
    load_top_movers,
    load_volume_vs_average,
)

st.set_page_config(
    page_title="Market intelligence — InvestIQ",
    page_icon="📊",
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
        --accent3:   #f59e0b;
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
    .mi-hero { padding: 1.5rem 0 0.75rem; }
    .mi-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em;
        text-transform: uppercase; color: var(--accent);
        background: rgba(0,229,180,0.08); border: 1px solid rgba(0,229,180,0.25);
        border-radius: 4px; padding: 0.3rem 0.75rem; margin-bottom: 0.75rem;
    }
    .mi-title {
        font-family: "Syne", sans-serif;
        font-size: clamp(2rem, 5vw, 3rem); font-weight: 800;
        letter-spacing: -0.04em; margin: 0 0 0.5rem;
        background: linear-gradient(135deg, #f0f4f8 30%, #00e5b4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .mi-sub { font-size: 0.95rem; color: var(--ink-muted); max-width: 44rem; line-height: 1.6; }

    /* ── Buttons ── */
    div.stButton > button[kind="secondary"] {
        border-radius: 7px !important; border: 1px solid var(--border2) !important;
        background: rgba(17,24,39,0.6) !important; color: var(--ink-muted) !important;
        font-weight: 500 !important; font-size: 0.85rem !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: var(--accent) !important; color: var(--accent) !important;
        background: rgba(0,229,180,0.05) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; }
    .stTabs [data-baseweb="tab"] { color: var(--ink-muted) !important; font-weight: 500 !important; }
    .stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25) !important;
        overflow: hidden !important;
    }
    [data-testid="stDataFrame"] [data-testid="stVerticalBlock"] > div {
        background: rgba(13,18,32,0.4) !important;
    }

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
render_navbar(current="marketiq")

st.markdown(
    """
    <div class="mi-hero">
        <div class="mi-badge">● Market Data</div>
        <div class="mi-title">Market intelligence</div>
    </div>
    """,
    unsafe_allow_html=True,
)

def _run_market_intel_loaders(result: dict, error: dict) -> None:
    """Runs the existing loader functions, untouched, on a background thread."""
    try:
        result["sectors"] = load_sector_benchmarks()
        result["top"] = load_top_movers(15, losers=False)
        result["bottom"] = load_top_movers(15, losers=True)
        result["vol"] = load_volume_vs_average(20)
        result["filings"] = load_recent_sec_filings(35)
        result["divs"] = load_recent_dividend_events(40)
        result["revenue"] = load_statement_highlights(25)
    except Exception as exc:  # noqa: BLE001 - surfaced to the main thread below
        error["exc"] = exc


# (progress %, status message) checkpoints — purely cosmetic, mirrors the
# real sequence of work happening inside the loader functions above.
_MI_LOAD_STAGES = [
    (0.05, "📊 Loading sector benchmarks..."),
    (0.35, "📈 Finding top movers & liquidity..."),
    (0.65, "📰 Pulling SEC filings & dividend events..."),
    (0.88, "💰 Loading revenue highlights..."),
    (0.97, "✨ Almost there..."),
]
# Typical observed load time — used only to pace the progress bar so it feels
# proportional to real elapsed time. The bar will still wait for the actual
# loaders to finish even if this estimate is off in either direction.
_MI_EXPECTED_LOAD_SECONDS = 14.0

_mi_progress_bar = st.empty()
_mi_status_text = st.empty()

_mi_result: dict = {}
_mi_error: dict = {}
_mi_load_thread = threading.Thread(
    target=_run_market_intel_loaders, args=(_mi_result, _mi_error), daemon=True
)
_mi_start_time = time.time()
_mi_load_thread.start()

_mi_stage_idx = 0
while _mi_load_thread.is_alive():
    _mi_elapsed = time.time() - _mi_start_time
    # Asymptotic curve: climbs quickly at first, then eases off, capped at 97%
    # until the real data is actually ready — so the bar never lies by hitting
    # 100% before the work is done.
    _mi_pct = min(0.97, 1 - math.exp(-_mi_elapsed / (_MI_EXPECTED_LOAD_SECONDS / 2.5)))
    while _mi_stage_idx < len(_MI_LOAD_STAGES) - 1 and _mi_pct >= _MI_LOAD_STAGES[_mi_stage_idx][0]:
        _mi_stage_idx += 1
    _mi_progress_bar.progress(_mi_pct, text=f"{int(_mi_pct * 100)}%")
    _mi_status_text.markdown(f"**{_MI_LOAD_STAGES[_mi_stage_idx][1]}**")
    time.sleep(0.12)

_mi_load_thread.join()

if "exc" in _mi_error:
    _mi_progress_bar.empty()
    _mi_status_text.empty()
    st.error(f"Error loading market data: {_mi_error['exc']}")
    st.info("Please ensure you have a stable internet connection for data fetching.")
    st.stop()

_mi_progress_bar.progress(1.0, text="100%")
_mi_status_text.markdown("**✅ Data loaded!**")
time.sleep(0.25)
_mi_progress_bar.empty()
_mi_status_text.empty()

sectors = _mi_result["sectors"]
top = _mi_result["top"]
bottom = _mi_result["bottom"]
vol = _mi_result["vol"]
filings = _mi_result["filings"]
divs = _mi_result["divs"]
revenue = _mi_result["revenue"]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Sectors",
    "Movers",
    "Volume",
    "SEC filings",
    "Dividends",
    "Revenue leaders",
])

with tab1:
    st.caption(
        "**Sectors:** 1Y momentum = percent change from the **latest** close to the "
        "**most recent close on or before** 365 calendar days ago (per symbol). "
        "**90d vol** = average daily range `(high-low)/close` as % over the last 90 "
        "calendar days (not annualized volatility)."
    )
    if sectors.empty:
        st.info("No sector aggregates returned.")
    else:
        sectors_view = sectors.copy()
        sectors_view["avg_1y_momentum_pct"] = pd.to_numeric(
            sectors_view["avg_1y_momentum_pct"], errors="coerce"
        )
        sectors_view["avg_90d_volatility_pct"] = pd.to_numeric(
            sectors_view["avg_90d_volatility_pct"], errors="coerce"
        )
        sectors_view["symbol_count"] = pd.to_numeric(
            sectors_view["symbol_count"], errors="coerce"
        )

        c1, c2 = st.columns([1.2, 1])
        with c1:
            sectors_display = sectors_view.rename(
                columns={
                    "sector": "Sector",
                    "symbol_count": "Stocks",
                    "avg_1y_momentum_pct": "Typical 1-year return (%)",
                    "avg_90d_volatility_pct": "Typical daily swing (%)",
                }
            )
            st.dataframe(
                sectors_display,
                column_config=_sectors_column_config(),
                height=360,
                **_MI_DF_KWARGS,
            )
        with c2:
            chart_df = (
                sectors_view.dropna(subset=["avg_1y_momentum_pct", "sector"])
                .sort_values("symbol_count", ascending=False)
                .head(12)
            )
            if chart_df.empty:
                st.info("No valid 1Y momentum values available for sector chart.")
            else:
                fig = px.bar(
                    chart_df,
                    x="avg_1y_momentum_pct",
                    y="sector",
                    orientation="h",
                    title="Typical one-year return by sector (top 12 by count)",
                    labels={
                        "avg_1y_momentum_pct": "Typical return (%)",
                        "sector": "Sector",
                    },
                    color_discrete_sequence=["#00e5b4"],
                )
                fig.update_layout(
                    margin=dict(l=8, r=8, t=40, b=8), height=360,
                    template="plotly_dark",
                    paper_bgcolor="rgba(13,18,32,0.85)",
                    plot_bgcolor="rgba(13,18,32,0.6)",
                    font=dict(color="#94a3b8"),
                    title_font=dict(color="#f0f4f8", family="Syne"),
                    xaxis=dict(gridcolor="rgba(99,202,183,0.08)", tickfont=dict(color="#475569")),
                    yaxis=dict(gridcolor="rgba(99,202,183,0.08)", tickfont=dict(color="#94a3b8")),
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.caption(
        "Sorted by the same **~1Y** price change as elsewhere: latest close vs last "
        "available close on or before **one year ago** (needs sufficient history in "
        "`stock_prices`)."
    )
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("##### Top 1Y momentum")
        st.dataframe(
            _rename_movers(top),
            column_config=_movers_column_config(),
            height=400,
            **_MI_DF_KWARGS,
        )
    with mc2:
        st.markdown("##### Bottom 1Y momentum")
        st.dataframe(
            _rename_movers(bottom),
            column_config=_movers_column_config(),
            height=400,
            **_MI_DF_KWARGS,
        )

with tab3:
    st.markdown("##### Volume vs prior 30-session average")
    st.caption(
        "Ratio = latest **trading day** volume ÷ mean volume of the **prior 30 trading "
        "days** for that symbol (latest day excluded from the mean)."
    )
    st.dataframe(
        _rename_volume(vol),
        column_config=_volume_column_config(),
        height=420,
        **_MI_DF_KWARGS,
    )

with tab4:
    if filings.empty:
        st.info("No SEC filing rows.")
    else:
        st.dataframe(
            _rename_filings(filings),
            column_config=_filings_column_config(),
            height=440,
            **_MI_DF_KWARGS,
        )

with tab5:
    if divs.empty:
        st.info("No dividend events.")
    else:
        st.dataframe(
            _rename_dividends(divs),
            column_config=_dividends_column_config(),
            height=440,
            **_MI_DF_KWARGS,
        )

with tab6:
    if revenue.empty:
        st.info("No statement rows matched revenue filters.")
    else:
        st.dataframe(
            _rename_revenue(revenue),
            column_config=_revenue_column_config(),
            height=440,
            **_MI_DF_KWARGS,
        )

st.divider()