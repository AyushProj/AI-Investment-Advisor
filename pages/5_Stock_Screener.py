"""Stock Screener — filter stocks by sector, price, momentum, and more."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from investiq_data import is_databricks_auth_failure, load_screener_stocks

st.set_page_config(
    page_title="Stock Screener — InvestIQ",
    page_icon="🔬",
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
        max-width: 84rem !important;
        padding-top: 1rem !important; padding-bottom: 4rem !important;
        padding-left: clamp(1rem, 3vw, 2.5rem) !important;
        padding-right: clamp(1rem, 3vw, 2.5rem) !important;
    }
    [data-testid="stSidebar"] { display: none !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

    .sc-hero { padding: 1.5rem 0 1rem; }
    .sc-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em;
        text-transform: uppercase; color: var(--accent);
        background: rgba(0,229,180,0.08); border: 1px solid rgba(0,229,180,0.25);
        border-radius: 4px; padding: 0.3rem 0.75rem; margin-bottom: 0.75rem;
    }
    .sc-title {
        font-family: "Syne", sans-serif;
        font-size: clamp(2rem, 5vw, 3rem); font-weight: 800;
        letter-spacing: -0.04em; margin: 0 0 0.5rem;
        background: linear-gradient(135deg, #f0f4f8 30%, #00e5b4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sc-sub { font-size: 0.9rem; color: var(--ink-muted); max-width: 50rem; line-height: 1.6; }

    .filter-panel {
        background: rgba(17,24,39,0.7);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.4rem 1.6rem 1rem;
        margin-bottom: 1.5rem;
    }

    .stat-strip { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.25rem; }
    .stat-chip {
        background: rgba(17,24,39,0.75); border: 1px solid var(--border);
        border-radius: 10px; padding: 0.55rem 1.1rem;
        font-size: 0.8rem; color: var(--ink-muted);
    }
    .stat-chip strong { color: var(--accent); font-size: 1rem; display: block; }

    /* ── Stock row ── */
    .stock-sym {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.95rem; font-weight: 700; color: var(--accent);
        letter-spacing: 0.04em;
    }
    .stock-name {
        font-size: 0.8rem; color: var(--ink-muted);
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        max-width: 16rem;
    }
    .stock-sector {
        font-size: 0.68rem; color: var(--ink-dim);
        font-family: "IBM Plex Mono", monospace; letter-spacing: 0.05em;
    }
    .stock-stat { font-size: 0.82rem; color: var(--ink); font-weight: 500; }
    .badge-bull {
        display: inline-block; padding: 0.18rem 0.55rem; border-radius: 5px;
        font-size: 0.7rem; font-weight: 600;
        background: rgba(0,229,180,0.12); color: #00e5b4;
        border: 1px solid rgba(0,229,180,0.3);
    }
    .badge-bear {
        display: inline-block; padding: 0.18rem 0.55rem; border-radius: 5px;
        font-size: 0.7rem; font-weight: 600;
        background: rgba(239,68,68,0.12); color: #f87171;
        border: 1px solid rgba(239,68,68,0.3);
    }
    .badge-neutral {
        display: inline-block; padding: 0.18rem 0.55rem; border-radius: 5px;
        font-size: 0.7rem; font-weight: 600;
        background: rgba(148,163,184,0.1); color: #94a3b8;
        border: 1px solid rgba(148,163,184,0.2);
    }
    /* ── Sticky column header bar ── */
    #screener-sticky-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(8, 12, 20, 0.97);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-bottom: 2px solid rgba(0, 229, 180, 0.4);
        box-shadow: 0 6px 28px rgba(0, 0, 0, 0.6);
        padding: 0.8rem 0;
        margin-bottom: 0.6rem;
    }
    .col-header {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #ffffff !important;
        text-shadow: 0 0 14px rgba(0, 229, 180, 0.4);
        padding-bottom: 0;
        border-bottom: none;
        margin-bottom: 0;
        white-space: nowrap;
    }

    /* Buttons */
    div.stButton > button[kind="primary"] {
        border-radius: 8px !important; border: none !important;
        background: linear-gradient(135deg, #00e5b4 0%, #00c49a 100%) !important;
        color: #080c14 !important; font-weight: 700 !important;
        font-family: "Syne", sans-serif !important;
        box-shadow: 0 8px 30px rgba(0,229,180,0.2) !important;
    }
    div.stButton > button[kind="secondary"] {
        border-radius: 7px !important; border: 1px solid var(--border2) !important;
        background: rgba(17,24,39,0.6) !important; color: var(--ink-muted) !important;
        font-weight: 500 !important; font-size: 0.8rem !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: var(--accent) !important; color: var(--accent) !important;
        background: rgba(0,229,180,0.06) !important;
    }

    .stSlider > div > div > div > div { background: var(--accent) !important; }
    [data-testid="stSliderThumb"] {
        background: var(--accent) !important; border: 2px solid #080c14 !important;
        box-shadow: 0 0 10px rgba(0,229,180,0.5) !important;
    }
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: var(--bg3) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important; color: var(--ink) !important;
    }
    .block-container label, .stSlider label, .stSelectbox label,
    .stMultiSelect label, .stNumberInput label {
        color: var(--ink-muted) !important; font-size: 0.78rem !important;
        font-weight: 500 !important; letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }
    [data-testid="stAlert"] {
        border-radius: 8px !important; border: 1px solid var(--border) !important;
        background: var(--bg3) !important;
    }
    hr { border: none !important; border-top: 1px solid var(--border) !important; }
    .row-divider {
        border: none; border-top: 1px solid rgba(99,202,183,0.07);
        margin: 0.1rem 0 0.3rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

from theme import inject_theme, render_navbar  # noqa: E402
inject_theme()
render_navbar(current="screener")

st.markdown(
    """
    <div class="sc-hero">
        <div class="sc-badge">● Databricks SQL warehouse</div>
        <div class="sc-title">Stock screener</div>
        <div class="sc-sub">
            Filter stocks by sector, price, momentum trend, EPS, dividends,
            volatility and revenue growth — all pulled live from Databricks.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load all data once (cached 5 min) ─────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _load() -> pd.DataFrame:
    return load_screener_stocks()


with st.spinner("Loading stocks from Databricks…"):
    try:
        df_all = _load()
    except Exception as exc:
        if is_databricks_auth_failure(exc):
            st.error("Databricks authentication failed.")
            st.markdown(
                "Use `DATABRICKS_HOST` + OAuth (`DATABRICKS_CLIENT_*`) or "
                "`DATABRICKS_CONFIG_PROFILE` after `databricks auth login`."
            )
            st.stop()
        raise

if df_all.empty:
    st.info("No stock data returned from Databricks.")
    st.stop()

# ── Normalise column types ─────────────────────────────────────────────────
for _c in ["last_close", "price_momentum", "volatility",
           "pays_dividends", "tailing_eps", "revenue_growth"]:
    if _c in df_all.columns:
        df_all[_c] = pd.to_numeric(df_all[_c], errors="coerce")

# ── Filter panel ──────────────────────────────────────────────────────────
st.markdown('<div class="filter-panel">', unsafe_allow_html=True)

r1c1, r1c2, r1c3 = st.columns([2, 1.5, 1.5])

with r1c1:
    sectors_available = (
        sorted(df_all["sector"].dropna().unique().tolist())
        if "sector" in df_all.columns else []
    )
    sectors_sel = st.multiselect(
        "Sector",
        options=sectors_available,
        placeholder="All sectors",
        help="Pick one or more sectors. Leave blank to include all.",
    )

with r1c2:
    trend_sel = st.selectbox(
        "Trend (1-year momentum)",
        options=[
            "All",
            "🟢 Bullish  (up > 10%)",
            "🔴 Bearish  (down > 10%)",
            "⚪ Neutral  (−10% to +10%)",
        ],
        help=(
            "Bullish = stock gained more than 10% over the last year.\n"
            "Bearish = dropped more than 10%.\n"
            "Neutral = within ±10%."
        ),
    )

with r1c3:
    dividend_filter = st.selectbox(
        "Dividend",
        options=["All", "Pays dividend", "No dividend"],
        help=(
            "Pays dividend = company paid a cash dividend in the last 2 years.\n"
            "No dividend = no dividend recorded."
        ),
    )

r2c1, r2c2, r2c3 = st.columns(3)

with r2c1:
    if "last_close" in df_all.columns and df_all["last_close"].notna().any():
        _p_max = min(float(df_all["last_close"].max(skipna=True)), 10_000.0)
    else:
        _p_max = 10_000.0

    price_max = st.slider(
        "Max share price (USD)",
        min_value=0.0,
        max_value=10_000.0,
        value=_p_max,
        step=50.0,
        format="$%.0f",
        help="Show only stocks whose last closing price is at or below this value.",
    )

with r2c2:
    volatility_filter = st.selectbox(
        "Volatility (90-day avg daily range)",
        options=[
            "All",
            "Low — steady  (< 1%)",
            "Medium  (1% – 3%)",
            "High — volatile  (> 3%)",
        ],
        help=(
            "Average daily high-low range as % of close over the last 90 days.\n"
            "Low = stable stock. High = big daily price swings."
        ),
    )

with r2c3:
    eps_filter = st.selectbox(
        "Earnings (trailing EPS)",
        options=["All", "Profitable  (EPS > 0)", "Loss-making  (EPS < 0)"],
        help=(
            "Trailing EPS = earnings per share over the last 12 months.\n"
            "Profitable = company is making money. Loss-making = currently at a loss."
        ),
    )

r3c1, r3c2, _ = st.columns(3)

with r3c1:
    growth_filter = st.selectbox(
        "Revenue growth (year-over-year)",
        options=[
            "All",
            "Strong growth  (> 20%)",
            "Positive growth  (> 0%)",
            "Declining  (< 0%)",
        ],
        help="How much the company's annual revenue changed compared to the prior year.",
    )

with r3c2:
    momentum_min = st.slider(
        "Min 1-year momentum %",
        min_value=-100,
        max_value=200,
        value=-100,
        step=5,
        format="%d%%",
        help=(
            "Floor on how much a stock must have moved over the last year.\n"
            "E.g. set to 0 to see only stocks that are up year-over-year."
        ),
    )

st.markdown("</div>", unsafe_allow_html=True)

# ── Apply filters in-memory ───────────────────────────────────────────────
df = df_all.copy()

if sectors_sel:
    df = df[df["sector"].isin(sectors_sel)]

if "price_momentum" in df.columns:
    if trend_sel.startswith("🟢"):
        df = df[df["price_momentum"] > 10]
    elif trend_sel.startswith("🔴"):
        df = df[df["price_momentum"] < -10]
    elif trend_sel.startswith("⚪"):
        df = df[df["price_momentum"].between(-10, 10, inclusive="both")]

if "price_momentum" in df.columns and momentum_min > -100:
    df = df[df["price_momentum"] >= momentum_min]

if "last_close" in df.columns:
    df = df[df["last_close"] <= price_max]

if "pays_dividends" in df.columns:
    if dividend_filter == "Pays dividend":
        df = df[df["pays_dividends"] == 1]
    elif dividend_filter == "No dividend":
        df = df[df["pays_dividends"] == 0]

if "volatility" in df.columns:
    if volatility_filter.startswith("Low"):
        df = df[df["volatility"] < 1]
    elif volatility_filter.startswith("Medium"):
        df = df[df["volatility"].between(1, 3, inclusive="both")]
    elif volatility_filter.startswith("High"):
        df = df[df["volatility"] > 3]

if "tailing_eps" in df.columns:
    if eps_filter.startswith("Profitable"):
        df = df[df["tailing_eps"] > 0]
    elif eps_filter.startswith("Loss"):
        df = df[df["tailing_eps"] < 0]

if "revenue_growth" in df.columns:
    if growth_filter.startswith("Strong"):
        df = df[df["revenue_growth"] > 20]
    elif growth_filter.startswith("Positive"):
        df = df[df["revenue_growth"] > 0]
    elif growth_filter.startswith("Declining"):
        df = df[df["revenue_growth"] < 0]

# ── Summary stat chips ────────────────────────────────────────────────────
total      = len(df)
bull_count = int((df["price_momentum"] > 10).sum())  if "price_momentum" in df.columns else 0
bear_count = int((df["price_momentum"] < -10).sum()) if "price_momentum" in df.columns else 0
div_count  = int((df["pays_dividends"] == 1).sum())  if "pays_dividends" in df.columns else 0
avg_eps    = df["tailing_eps"].dropna().mean()        if "tailing_eps"    in df.columns else None

chips_html = f"""
<div class="stat-strip">
  <div class="stat-chip"><strong>{total:,}</strong>Stocks matched</div>
  <div class="stat-chip"><strong style="color:#00e5b4">{bull_count:,}</strong>Bullish</div>
  <div class="stat-chip"><strong style="color:#f87171">{bear_count:,}</strong>Bearish</div>
  <div class="stat-chip"><strong style="color:#f59e0b">{div_count:,}</strong>Pay dividends</div>
"""
if avg_eps is not None and not pd.isna(avg_eps):
    chips_html += f'<div class="stat-chip"><strong>${avg_eps:.2f}</strong>Avg EPS</div>'
chips_html += "</div>"
st.markdown(chips_html, unsafe_allow_html=True)

# ── Results list ──────────────────────────────────────────────────────────
if df.empty:
    st.info("No stocks match your current filters. Try relaxing some criteria.")
else:
    df = df.reset_index(drop=True)

    # ── Helper formatters ──────────────────────────────────────────────────
    def _fmt_price(v) -> str:
        try:
            return f"${float(v):,.2f}" if pd.notna(v) else "—"
        except (TypeError, ValueError):
            return "—"

    def _fmt_pct(v, decimals: int = 1) -> str:
        try:
            return f"{float(v):+.{decimals}f}%" if pd.notna(v) else "—"
        except (TypeError, ValueError):
            return "—"

    def _fmt_eps(v) -> str:
        try:
            return f"${float(v):.2f}" if pd.notna(v) else "—"
        except (TypeError, ValueError):
            return "—"

    def _trend_badge(v) -> str:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return '<span class="badge-neutral">—</span>'
        if fv > 10:
            return '<span class="badge-bull">🟢 Bullish</span>'
        if fv < -10:
            return '<span class="badge-bear">🔴 Bearish</span>'
        return '<span class="badge-neutral">⚪ Neutral</span>'

    # ── Sticky column header — rendered as a single HTML block so
    #    position:sticky works relative to the page scroll container.
    # ──────────────────────────────────────────────────────────────────────
    _COLS = [1.6, 1, 1.1, 1, 1, 1, 0.85]
    _HDR_LABELS = ["Company", "Last Price", "1Y Momentum",
                   "90d Volatility", "Trailing EPS", "Rev Growth", "Details"]

    _total_flex = sum(_COLS)
    _hdr_cells = "".join(
        f'<div style="flex:{w/_total_flex:.4f};min-width:0;" class="col-header">{lbl}</div>'
        for w, lbl in zip(_COLS, _HDR_LABELS)
    )

    st.markdown(
        f"""
        <div id="screener-sticky-header">
          <div style="display:flex;gap:1rem;align-items:center;">
            {_hdr_cells}
          </div>
        </div>
        <script>
        (function() {{
            function fixSticky() {{
                var hdr = document.getElementById('screener-sticky-header');
                if (!hdr) return;
                var scroller = hdr;
                for (var i = 0; i < 20; i++) {{
                    scroller = scroller.parentElement;
                    if (!scroller) break;
                    var overflow = window.getComputedStyle(scroller).overflowY;
                    if (overflow === 'auto' || overflow === 'scroll') {{
                        scroller.insertBefore(hdr, scroller.firstChild);
                        hdr.style.paddingLeft = '2.5rem';
                        hdr.style.paddingRight = '2.5rem';
                        break;
                    }}
                }}
            }}
            if (document.readyState === 'complete') {{ fixSticky(); }}
            else {{ window.addEventListener('load', fixSticky); }}
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

    # ── One row per stock ──────────────────────────────────────────────────
    for _, row in df.iterrows():
        sym      = str(row.get("symbol", ""))
        name     = str(row.get("company_name", sym))
        sector   = str(row.get("sector", ""))
        div_icon = "💰 " if int(row.get("pays_dividends", 0) or 0) == 1 else ""

        c_sym, c_price, c_mom, c_vol, c_eps, c_rev, c_btn = st.columns(_COLS)

        # Company identity
        c_sym.markdown(
            f"""
            <div style="padding:0.15rem 0 0.1rem;">
                <div class="stock-sym">{div_icon}{sym}</div>
                <div class="stock-name" title="{name}">{name}</div>
                <div class="stock-sector">{sector}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metrics
        c_price.markdown(
            f'<div style="padding-top:0.3rem;" class="stock-stat">{_fmt_price(row.get("last_close"))}</div>',
            unsafe_allow_html=True,
        )
        mom_val = row.get("price_momentum")
        c_mom.markdown(
            f'<div style="padding-top:0.2rem;">{_trend_badge(mom_val)}'
            f'<br><span style="font-size:0.74rem;color:var(--ink-muted);">{_fmt_pct(mom_val)}</span></div>',
            unsafe_allow_html=True,
        )
        c_vol.markdown(
            f'<div style="padding-top:0.3rem;" class="stock-stat">{_fmt_pct(row.get("volatility"), 2)}</div>',
            unsafe_allow_html=True,
        )
        c_eps.markdown(
            f'<div style="padding-top:0.3rem;" class="stock-stat">{_fmt_eps(row.get("tailing_eps"))}</div>',
            unsafe_allow_html=True,
        )
        c_rev.markdown(
            f'<div style="padding-top:0.3rem;" class="stock-stat">{_fmt_pct(row.get("revenue_growth"))}</div>',
            unsafe_allow_html=True,
        )

        # View Details button — stores selected symbol in session_state,
        # then switches to the Company Details page which reads it.
        with c_btn:
            st.markdown('<div style="padding-top:0.15rem;">', unsafe_allow_html=True)
            if st.button("View →", key=f"view_{sym}", type="secondary"):
                st.session_state["selected_symbol"] = sym
                st.switch_page("pages/1_Company_Details.py")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<hr class="row-divider">', unsafe_allow_html=True)

st.divider()