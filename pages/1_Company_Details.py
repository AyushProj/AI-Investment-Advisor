from datetime import timedelta
import html
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from price_forecast_ml import (
    arima_forecast_confidence,
    build_forecast_chart_figure,
    run_rf_signal,
    run_rf_regression,      # Item 4
    get_sentiment_score,    # Item 2
)
from investiq_data import SQL_WAREHOUSE_ID, _tbl
from llm_chat import chat_completion, get_gemini_api_key, get_gemini_model
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import time

st.set_page_config(page_title="Company Details", page_icon="📊", layout="wide")

WAREHOUSE_ID = SQL_WAREHOUSE_ID
CHUNK_SIZE = 1000


def _run_query(w: WorkspaceClient, sql: str, label: str) -> pd.DataFrame:
    all_rows = []
    columns = None
    offset = 0
    try:
        while True:
            paginated_sql = f"SELECT * FROM ({sql}) _q LIMIT {CHUNK_SIZE} OFFSET {offset}"
            resp = w.statement_execution.execute_statement(
                warehouse_id=WAREHOUSE_ID, statement=paginated_sql, wait_timeout="30s",
            )
            while resp.status.state in [StatementState.PENDING, StatementState.RUNNING]:
                time.sleep(1)
                resp = w.statement_execution.get_statement(resp.statement_id)
            if resp.status.state != StatementState.SUCCEEDED:
                st.error(f"Query '{label}' failed: {getattr(resp.status, 'error', 'unknown')}")
                break
            if columns is None:
                schema = None
                if resp.manifest is not None and hasattr(resp.manifest, "schema") and resp.manifest.schema is not None:
                    schema = resp.manifest.schema
                elif hasattr(resp.result, "schema") and resp.result.schema is not None:
                    schema = resp.result.schema
                if schema is None:
                    st.error(f"Query '{label}': could not locate schema.")
                    return pd.DataFrame()
                columns = [c.name for c in schema.columns]
            if resp.result is None or not resp.result.data_array:
                break
            chunk = resp.result.data_array
            all_rows.extend(chunk)
            if len(chunk) < CHUNK_SIZE:
                break
            offset += CHUNK_SIZE
        if not all_rows:
            return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
        return pd.DataFrame(all_rows, columns=columns)
    except Exception as exc:
        st.error(f"Exception running query '{label}': {exc}")
        return pd.DataFrame()


@st.cache_data
def load_company_prices(symbol: str) -> pd.DataFrame:
    w = WorkspaceClient()
    query = f"""
        SELECT symbol, report_date, open, close, high, low, volume
        FROM {_tbl('stock_prices')}
        WHERE UPPER(TRIM(symbol)) = '{symbol}'
        ORDER BY report_date
    """
    df = _run_query(w, query, f"prices_{symbol}")
    if df.empty:
        return pd.DataFrame()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["report_date", "close"])
    return df.sort_values("report_date").reset_index(drop=True)


@st.cache_data
def load_company_profile(symbol: str) -> dict | None:
    w = WorkspaceClient()
    query = f"""
        SELECT symbol, sector, industry, long_business_summary
        FROM {_tbl('stock_profile')}
        WHERE UPPER(TRIM(symbol)) = '{symbol}'
        ORDER BY report_date DESC
        LIMIT 1
    """
    df = _run_query(w, query, f"profile_{symbol}")
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "sector":   row.get("sector", ""),
        "industry": row.get("industry", ""),
        "summary":  row.get("long_business_summary", ""),
    }


def get_llm_response(symbol, current_price, current_date, profile, user_prompt):
    model_name = get_gemini_model()
    if not get_gemini_api_key():
        return "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys"
    try:
        sector   = profile.get("sector",   "N/A") if profile else "N/A"
        industry = profile.get("industry", "N/A") if profile else "N/A"
        summary  = profile.get("summary",  "No summary available.") if profile else "No summary available."
        system_prompt = f"""You are an AI investment assistant for a student hackathon project.
You are a decision-support assistant, not a trading bot.
Do not guarantee returns. Do not say "buy this now" or "sell immediately."
Be practical, cautious, and explainable.

Company context:
Symbol: {symbol}
Latest closing price: ${current_price:,.2f}
Latest available date: {current_date.strftime("%Y-%m-%d")}
Sector: {sector} | Industry: {industry}
Company summary: {summary}
"""
        return chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"User question about {symbol}: {user_prompt}"},
            ],
            model=model_name, temperature=0.3,
        )
    except Exception as e:
        return f"LLM request failed: {e}"


def go_home():
    st.session_state.selected_symbol = None
    st.switch_page("app.py")


if "selected_symbol" not in st.session_state or not st.session_state.selected_symbol:
    st.warning("No company selected. Please go back and choose a company.")
    if st.button("Go to Home"):
        go_home()
    st.stop()

symbol = st.session_state.selected_symbol

with st.spinner(f"Loading data for {symbol}..."):
    price_df = load_company_prices(symbol)
    profile  = load_company_profile(symbol)

if price_df.empty:
    st.error(f"No price data found for {symbol}.")
    if st.button("Back to Home"):
        go_home()
    st.stop()

latest_row    = price_df.iloc[-1]
current_price = latest_row["close"]
current_date  = latest_row["report_date"]

# ── STYLES ────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg:#080c14;--bg2:#0d1220;--bg3:#111827;
        --surface:rgba(17,24,39,0.85);--surface2:rgba(30,41,59,0.7);
        --border:rgba(99,202,183,0.12);--border2:rgba(99,202,183,0.22);
        --accent:#00e5b4;--accent2:#3b82f6;--accent3:#f59e0b;
        --ink:#f0f4f8;--ink-muted:#94a3b8;--ink-dim:#475569;
        --shadow:0 20px 60px rgba(0,0,0,0.5);--shadow-sm:0 4px 20px rgba(0,0,0,0.35);
        --glow:0 0 30px rgba(0,229,180,0.12);
    }
    .stApp {
        font-family:"Inter",system-ui,sans-serif !important;
        background:
            radial-gradient(ellipse 80vw 60vh at 15% -10%,rgba(0,229,180,0.07) 0%,transparent 60%),
            radial-gradient(ellipse 60vw 50vh at 85% 5%,rgba(59,130,246,0.06) 0%,transparent 55%),
            linear-gradient(180deg,#080c14 0%,#0a0f1c 50%,#080c14 100%) !important;
        color:var(--ink) !important;
    }
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width:80rem !important;padding-top:1rem !important;padding-bottom:4rem !important;
        padding-left:clamp(1rem,3vw,2.5rem) !important;padding-right:clamp(1rem,3vw,2.5rem) !important;
    }
    [data-testid="stSidebar"]{display:none !important;}
    ::-webkit-scrollbar{width:6px;}
    ::-webkit-scrollbar-track{background:var(--bg);}
    ::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px;}
    .detail-hero{margin-bottom:1.5rem;padding-top:1.5rem;}
    .detail-hero .symbol-pill{
        display:inline-flex;align-items:center;gap:0.4rem;
        font-family:"IBM Plex Mono",monospace;font-size:0.68rem;font-weight:600;
        letter-spacing:0.15em;text-transform:uppercase;color:var(--accent);
        background:rgba(0,229,180,0.08);border:1px solid rgba(0,229,180,0.25);
        border-radius:4px;padding:0.3rem 0.75rem;margin-bottom:0.75rem;
    }
    .detail-hero h1{
        font-family:"Syne",sans-serif;font-size:clamp(1.6rem,3vw,2.2rem);font-weight:800;
        letter-spacing:-0.04em;margin:0 0 0.4rem;
        background:linear-gradient(135deg,#f0f4f8 30%,#00e5b4 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    }
    .detail-hero .sub{color:var(--ink-muted);font-size:0.95rem;margin-top:0.25rem;}
    .panel{
        background:var(--surface);backdrop-filter:blur(16px);border-radius:16px;
        padding:1.25rem 1.25rem 0.75rem;border:1px solid var(--border);
        box-shadow:var(--shadow),var(--glow);margin-bottom:1rem;position:relative;overflow:hidden;
    }
    .panel::before{
        content:"";position:absolute;top:0;left:0;right:0;height:2px;
        background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:0.5;
    }
    .summary-card{
        background:rgba(0,0,0,0.25);border-radius:10px;padding:1rem 1.1rem;
        border:1px solid var(--border);line-height:1.65;color:var(--ink-muted);font-size:0.9rem;
    }
    .chat-aside h3{font-family:"Syne",sans-serif;font-size:1.05rem;font-weight:700;color:var(--ink);margin-bottom:0.25rem;}
    .chat-aside .muted{font-size:0.85rem;color:var(--ink-muted);margin-bottom:0.75rem;}
    div.stButton > button[kind="primary"]{
        border-radius:8px !important;border:none !important;
        background:linear-gradient(135deg,#00e5b4 0%,#00c49a 100%) !important;
        color:#080c14 !important;font-weight:700 !important;font-family:"Syne",sans-serif !important;
    }
    div.stButton > button[kind="secondary"]{
        border-radius:7px !important;border:1px solid var(--border2) !important;
        background:rgba(17,24,39,0.6) !important;color:var(--ink-muted) !important;
        font-weight:500 !important;font-size:0.85rem !important;
    }
    div.stButton > button[kind="secondary"]:hover{
        border-color:var(--accent) !important;color:var(--accent) !important;
        background:rgba(0,229,180,0.05) !important;
    }
    [data-testid="stMetricValue"]{color:var(--accent) !important;font-family:"IBM Plex Mono",monospace !important;}
    [data-testid="stMetricLabel"]{color:var(--ink-dim) !important;font-size:0.75rem !important;}
    .stTabs [data-baseweb="tab-list"]{background:transparent !important;border-bottom:1px solid var(--border) !important;}
    .stTabs [data-baseweb="tab"]{color:var(--ink-muted) !important;font-weight:500 !important;}
    .stTabs [aria-selected="true"]{color:var(--accent) !important;border-bottom:2px solid var(--accent) !important;}
    .stSelectbox > div > div,.stNumberInput > div > div > input{
        background:var(--bg3) !important;border:1px solid var(--border) !important;
        border-radius:8px !important;color:var(--ink) !important;
    }
    .stSlider > div > div > div > div{background:var(--accent) !important;}
    [data-testid="stSliderThumb"]{
        background:var(--accent) !important;border:2px solid #080c14 !important;
        box-shadow:0 0 10px rgba(0,229,180,0.5) !important;
    }
    .block-container label,.stSelectbox label,.stSlider label,.stNumberInput label{
        color:var(--ink-muted) !important;font-size:0.8rem !important;
        font-weight:500 !important;letter-spacing:0.03em !important;text-transform:uppercase !important;
    }
    [data-testid="stExpander"]{
        background:var(--bg3) !important;border:1px solid var(--border) !important;border-radius:10px !important;
    }
    [data-testid="stAlert"]{
        border-radius:8px !important;border:1px solid var(--border) !important;background:var(--bg3) !important;
    }
    hr{border:none !important;border-top:1px solid var(--border) !important;}
</style>
""", unsafe_allow_html=True)

from theme import inject_theme, render_navbar  # noqa: E402
inject_theme()
render_navbar()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
    <div class="detail-hero">
        <div class="symbol-pill">● {html.escape(str(symbol))}</div>
        <h1>{html.escape(str(symbol))}</h1>
        <div class="sub">Price history, profile, and an AI assistant — not trading advice.</div>
    </div>
""", unsafe_allow_html=True)

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
left, right = st.columns([3.2, 1.3])

with left:
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"${current_price:,.2f}")
    c2.metric("Latest Date",   current_date.strftime("%Y-%m-%d"))
    c3.metric("Records",       f"{len(price_df):,}")

    tab_chart, tab_predictor, tab_rf = st.tabs([
        "Price chart", "ARIMA forecast", "🤖 ML Signal"
    ])

    # ── TAB 1: Price chart ────────────────────────────────────────────────────
    with tab_chart:
        range_option = st.selectbox(
            "Select chart range",
            ["1M", "1Y", "3Y", "5Y", "MAX", "Custom Years"],
            index=0, key="chart_range_sel",
        )
        filtered_df = price_df.copy()
        if range_option == "1M":
            filtered_df = price_df[price_df["report_date"] >= current_date - timedelta(days=30)]
        elif range_option == "1Y":
            filtered_df = price_df[price_df["report_date"] >= current_date - timedelta(days=365)]
        elif range_option == "3Y":
            filtered_df = price_df[price_df["report_date"] >= current_date - timedelta(days=365*3)]
        elif range_option == "5Y":
            filtered_df = price_df[price_df["report_date"] >= current_date - timedelta(days=365*5)]
        elif range_option == "Custom Years":
            years = st.number_input("Enter number of years", min_value=1, max_value=50, value=2, step=1, key="chart_custom_years")
            filtered_df = price_df[price_df["report_date"] >= current_date - timedelta(days=365*int(years))]

        fig = px.line(filtered_df, x="report_date", y="close",
                      title=f"{symbol} Closing Price",
                      labels={"report_date": "Date", "close": "Close Price"})
        fig.update_traces(mode="lines", line=dict(color="#00e5b4", width=2),
                          hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>")
        fig.update_layout(
            height=480, template="plotly_dark",
            paper_bgcolor="rgba(13,18,32,0.85)", plot_bgcolor="rgba(13,18,32,0.6)",
            margin=dict(l=20, r=20, t=50, b=20), font=dict(color="#94a3b8"),
            title_font=dict(color="#f0f4f8", family="Syne"),
            xaxis=dict(tickfont=dict(color="#475569"), gridcolor="rgba(99,202,183,0.08)"),
            yaxis=dict(tickfont=dict(color="#475569"), gridcolor="rgba(99,202,183,0.08)"),
        )
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: ARIMA forecast ─────────────────────────────────────────────────
    with tab_predictor:
        st.markdown("**ARIMA forecast + confidence band** (statsmodels)")
        st.caption("Uses recent daily closes. Not investment advice. Band = model uncertainty.")
        fc_horizon = st.slider("Forecast horizon (trading days)", 5, 60, 20, key="fc_horizon")
        fc_conf    = st.selectbox("Interval level", [0.90, 0.95, 0.99], index=1, key="fc_conf")
        run_fc     = st.button("Run forecast", type="primary", key="run_ml_forecast")
        train_df   = price_df.tail(800).copy()
        if run_fc:
            fc_result = arima_forecast_confidence(train_df, horizon=int(fc_horizon),
                                                   confidence=float(fc_conf), min_obs=50)
            # ── CHANGE 1: Better message (ARIMA now always returns something) ──
            if fc_result is None:
                st.warning("Could not build a forecast — need at least 50 clean price rows.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric(f"Forecast ({fc_horizon}d)", f"${fc_result.point[-1]:,.2f}")
                m2.metric("Band low",  f"${fc_result.lower[-1]:,.2f}")
                m3.metric("Band high", f"${fc_result.upper[-1]:,.2f}")
                # ── CHANGE 2: Show model type (ARIMA or linear trend fallback) ──
                is_linear = fc_result.model_order == ("linear trend",)
                model_label = (
                    "Linear trend fallback (ARIMA could not converge on this price series)"
                    if is_linear
                    else f"ARIMA{fc_result.model_order}"
                )
                st.caption(f"Model: {model_label} · trained on **{fc_result.n_obs}** daily closes.")
                st.plotly_chart(build_forecast_chart_figure(train_df, fc_result, symbol),
                                use_container_width=True)

    # ── TAB 3: ML Signal (RF + Sentiment + Regression) ───────────────────────
    with tab_rf:

        # ══════════════════════════════════════════════════════════════════════
        # SECTION A — Buy / Hold / Avoid Classification
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("### 🎯 Investment signal")
        st.caption("Random Forest trained on 14 technical features. "
                   "TimeSeriesSplit CV — never sees future data. Not investment advice.")

        run_rf_btn = st.button("Run ML signal analysis", type="primary", key="run_rf_btn")

        if run_rf_btn:
            with st.spinner("Training Random Forest classifier... 15–25 seconds"):
                rf = run_rf_signal(price_df)

            if rf is None:
                st.warning("Not enough data (need ≥150 rows). Try AAPL, MSFT, NVDA.")
            else:
                signal       = rf["signal"]
                signal_emoji = {"Buy":"📈","Hold":"⏸️","Avoid":"📉","Abstain":"🤔"}.get(signal,"🤔")
                signal_color = {"Buy":"#00e564","Hold":"#f59e0b","Avoid":"#ef4444","Abstain":"#94a3b8"}.get(signal,"#94a3b8")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ML Signal",    f"{signal_emoji} {signal}")
                m2.metric("Confidence",   f"{rf['confidence']*100:.0f}%")
                m3.metric("Risk Score",   f"{rf['risk_score']} / 10")
                m4.metric("CV Accuracy",  f"{rf['cv_accuracy']*100:.1f}%", f"±{rf['cv_std']*100:.1f}%")

                # ── CHANGE 3: Abstain warning uses low_confidence + top_gap ──
                if signal == "Abstain":
                    st.warning(
                        f"🤔 **Abstain** — the top two signals are tied "
                        f"(gap = {rf['top_gap']*100:.1f}%). "
                        "The model cannot confidently distinguish between them. "
                        "No clear recommendation — consider waiting for a clearer signal."
                    )

                st.markdown("---")
                left_rf, right_rf = st.columns([1, 1.5])

                with left_rf:
                    st.markdown("##### Signal probabilities")
                    prob_colors = {"Buy":"#00e564","Hold":"#f59e0b","Avoid":"#ef4444"}
                    for lbl in ["Buy", "Hold", "Avoid"]:
                        p = rf["probabilities"].get(lbl, 0)
                        st.markdown(f"""
                        <div style="margin-bottom:10px;">
                          <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;">
                            <span style="color:#94a3b8;">{lbl}</span>
                            <span style="color:{prob_colors[lbl]};font-weight:600;">{p*100:.0f}%</span>
                          </div>
                          <div style="background:rgba(255,255,255,0.08);border-radius:4px;height:7px;overflow:hidden;">
                            <div style="width:{p*100}%;background:{prob_colors[lbl]};height:100%;border-radius:4px;"></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("##### Current indicators")
                    for k, v in rf["latest"].items():
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'>"
                            f"<span style='color:#94a3b8;font-size:13px;'>{k}</span>"
                            f"<span style='color:#f0f4f8;font-size:13px;font-weight:600;'>{v}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                with right_rf:
                    st.markdown("##### Feature importance")
                    fi = rf["feat_imp"].head(10).copy()
                    labels_map = {
                        "mom_5d":"Momentum 5d","mom_20d":"Momentum 20d","mom_60d":"Momentum 60d",
                        "price_vs_sma20":"vs SMA-20","price_vs_sma50":"vs SMA-50","sma_cross":"SMA Cross",
                        "volatility_20d":"Volatility 20d","volatility_60d":"Volatility 60d",
                        "rsi_14":"RSI (14)","macd":"MACD","macd_signal":"MACD Signal",
                        "macd_hist":"MACD Histogram","vol_ratio":"Volume Ratio","pos_52w":"52w Position",
                    }
                    fi["label"] = fi["feature"].map(labels_map)
                    fi_pos = fi[fi["importance"] > 0]
                    if not fi_pos.empty:
                        fig_fi = px.bar(fi_pos, x="importance", y="label", orientation="h",
                                        color="importance", color_continuous_scale=["#1e3a5f","#00e5b4"],
                                        labels={"importance":"Importance","label":""})
                        fig_fi.update_layout(
                            template="plotly_dark", paper_bgcolor="rgba(13,18,32,0.85)",
                            plot_bgcolor="rgba(13,18,32,0.6)", font=dict(color="#94a3b8"),
                            height=340, margin=dict(l=8,r=8,t=8,b=8),
                            showlegend=False, coloraxis_showscale=False,
                            yaxis=dict(categoryorder="total ascending"),
                        )
                        st.plotly_chart(fig_fi, use_container_width=True)

                with st.expander("Model evaluation details"):
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Training samples", f"{rf['n_training']:,}")
                    e2.metric("CV folds",          "5 (time-series)")
                    e3.metric("CV accuracy",       f"{rf['cv_accuracy']*100:.1f}%")
                    e4.metric("Confidence threshold", "55%")
                    st.markdown("""
**Labeling** (forward 20-day return):
- **Buy** → > +4% · **Hold** → -2% to +4% · **Avoid** → < -2%

**Abstain rule:** when the top two signal probabilities are within 3% of each other the model is genuinely tied and abstains rather than guessing.

**No data leakage:** TimeSeriesSplit always trains on past, tests on future.
                    """)

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION B — NEWS SENTIMENT
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("### 📰 News sentiment analysis")
        st.caption("LLM-scored sentiment from recent Yahoo Finance headlines. "
                   "Detects if sentiment is improving or deteriorating.")

        run_sent_btn = st.button("Analyze news sentiment", type="primary", key="run_sent_btn")

        if run_sent_btn:
            with st.spinner("Fetching headlines and scoring sentiment..."):
                sent = get_sentiment_score(symbol)

            if sent.get("error") and sent["count"] == 0:
                st.warning(f"Could not fetch sentiment: {sent.get('error', 'unknown error')}")
            elif sent["count"] == 0:
                st.info("No news headlines found for this ticker.")
            else:
                sent_color = {
                    "Positive": "#00e564",
                    "Negative": "#ef4444",
                    "Neutral":  "#f59e0b",
                }.get(sent["label"], "#94a3b8")

                shift_color = {
                    "Improving ↑":     "#00e564",
                    "Deteriorating ↓": "#ef4444",
                    "Stable →":        "#f59e0b",
                }.get(sent["shift_label"], "#94a3b8")

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Sentiment Score",  f"{sent['score']:+.3f}")
                s2.metric("Overall Label",    sent["label"])
                s3.metric("Trend",            sent["shift_label"])
                s4.metric("Headlines scored", str(sent["count"]))

                st.markdown("---")
                sl, sr = st.columns([1, 1])

                with sl:
                    st.markdown("##### Sentiment breakdown")
                    for lbl, pct, color in [
                        ("Positive", sent["positive_pct"], "#00e564"),
                        ("Neutral",  sent["neutral_pct"],  "#f59e0b"),
                        ("Negative", sent["negative_pct"], "#ef4444"),
                    ]:
                        st.markdown(f"""
                        <div style="margin-bottom:10px;">
                          <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;">
                            <span style="color:#94a3b8;">{lbl}</span>
                            <span style="color:{color};font-weight:600;">{pct}%</span>
                          </div>
                          <div style="background:rgba(255,255,255,0.08);border-radius:4px;height:7px;overflow:hidden;">
                            <div style="width:{pct}%;background:{color};height:100%;border-radius:4px;"></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    if sent.get("summary"):
                        st.markdown(f"""
                        <div style="margin-top:12px;padding:10px 14px;background:rgba(0,0,0,0.2);
                                    border-radius:8px;border:1px solid rgba(99,202,183,0.15);">
                            <span style="color:#94a3b8;font-size:13px;font-style:italic;">
                                "{sent['summary']}"
                            </span>
                        </div>""", unsafe_allow_html=True)

                with sr:
                    st.markdown("##### Sentiment shift (recent vs older)")
                    st.markdown(f"""
                    <div style="padding:16px;background:rgba(0,0,0,0.2);border-radius:10px;
                                border:1px solid rgba(99,202,183,0.15);margin-bottom:12px;">
                        <div style="font-size:13px;color:#94a3b8;margin-bottom:8px;">
                            <strong style="color:#f0f4f8;">Recent headlines (1-7):</strong>
                            <span style="color:{sent_color};font-weight:600;margin-left:8px;">
                                {sent['recent_avg']:+.3f}
                            </span>
                        </div>
                        <div style="font-size:13px;color:#94a3b8;margin-bottom:8px;">
                            <strong style="color:#f0f4f8;">Older headlines (8-20):</strong>
                            <span style="color:#94a3b8;font-weight:600;margin-left:8px;">
                                {sent['older_avg']:+.3f}
                            </span>
                        </div>
                        <div style="font-size:15px;font-weight:700;color:{shift_color};margin-top:10px;">
                            {sent['shift_label']}
                            <span style="font-size:12px;color:#94a3b8;font-weight:400;margin-left:6px;">
                                (shift = {sent['shift']:+.3f})
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="font-size:11px;color:#475569;margin-top:8px;">
                        Model: {sent.get('model','—')}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION C — PRICE REGRESSION
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("### 📊 Price prediction (regression)")
        st.caption("Random Forest Regressor predicting actual price in 5 / 10 / 20 trading days. "
                   "Confidence interval from spread of 200 individual trees. Not investment advice.")

        run_reg_btn = st.button("Run price prediction", type="primary", key="run_reg_btn")

        if run_reg_btn:
            with st.spinner("Training regression model... 15–25 seconds"):
                reg = run_rf_regression(price_df)

            if reg is None:
                st.warning("Not enough data (need ≥150 rows).")
            else:
                st.markdown(f"**Current price:** `${reg['current_price']:,.2f}`")

                r1, r2, r3 = st.columns(3)
                cols_map = {5: r1, 10: r2, 20: r3}

                for horizon, col in cols_map.items():
                    if horizon not in reg["predictions"]:
                        continue
                    p = reg["predictions"][horizon]
                    ret = p["predicted_return"]
                    ret_color = "#00e564" if ret > 0 else "#ef4444"
                    mape_str  = f"{p['mape']:.1f}% MAPE" if p["mape"] else "N/A"

                    with col:
                        st.markdown(f"""
                        <div style="padding:16px;background:rgba(0,0,0,0.2);border-radius:12px;
                                    border:1px solid rgba(99,202,183,0.2);text-align:center;">
                            <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;
                                        letter-spacing:0.1em;margin-bottom:6px;">
                                {horizon}-day prediction
                            </div>
                            <div style="font-family:'IBM Plex Mono';font-size:1.4rem;
                                        font-weight:700;color:#f0f4f8;">
                                ${p['predicted_price']:,.2f}
                            </div>
                            <div style="font-size:12px;color:{ret_color};font-weight:600;margin:4px 0;">
                                {ret:+.1f}% predicted return
                            </div>
                            <div style="font-size:11px;color:#475569;">
                                Range: ${p['low']:,.2f} – ${p['high']:,.2f}
                            </div>
                            <div style="font-size:11px;color:#475569;margin-top:4px;">
                                Accuracy: {mape_str}
                            </div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("##### Predicted price range")
                horizons = sorted(reg["predictions"].keys())
                fig_reg = go.Figure()

                fig_reg.add_trace(go.Scatter(
                    x=[0], y=[reg["current_price"]],
                    mode="markers", name="Current price",
                    marker=dict(color="#00e5b4", size=12, symbol="circle"),
                ))

                pred_x = [h for h in horizons if h in reg["predictions"]]
                pred_y = [reg["predictions"][h]["predicted_price"] for h in pred_x]
                low_y  = [reg["predictions"][h]["low"]  for h in pred_x]
                high_y = [reg["predictions"][h]["high"] for h in pred_x]

                fig_reg.add_trace(go.Scatter(
                    x=pred_x + pred_x[::-1],
                    y=high_y + low_y[::-1],
                    fill="toself", fillcolor="rgba(0,229,180,0.1)",
                    line=dict(color="rgba(0,229,180,0)"),
                    name="95% confidence band",
                ))
                fig_reg.add_trace(go.Scatter(
                    x=pred_x, y=pred_y,
                    mode="lines+markers", name="Predicted price",
                    line=dict(color="#00e5b4", width=2, dash="dash"),
                    marker=dict(size=8),
                ))

                fig_reg.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(13,18,32,0.85)",
                    plot_bgcolor="rgba(13,18,32,0.6)",
                    font=dict(color="#94a3b8"), height=300,
                    margin=dict(l=8,r=8,t=30,b=8),
                    xaxis=dict(title="Trading days ahead", gridcolor="rgba(99,202,183,0.08)"),
                    yaxis=dict(title="Price (USD)",        gridcolor="rgba(99,202,183,0.08)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_reg, use_container_width=True)

                with st.expander("How to read this"):
                    st.markdown("""
**Point prediction** — the model's best guess for each horizon.

**Confidence band** — the spread of predictions from all 200 trees.
Wider band = model is less certain about that timeframe.

**MAPE (accuracy)** — measured on held-out future data:
"On average, predictions for this horizon were off by X%."
Lower is better. 5% = good. 15%+ = model struggling.

**Important:** predictions are based only on price patterns.
News, earnings, and macro events are not included.
                    """)

        st.caption("⚠️ All models use historical price data only. Not investment advice.")

    # ── Company summary ───────────────────────────────────────────────────────
    st.markdown("##### Company summary")
    if profile and profile.get("summary"):
        st.markdown(f'<div class="summary-card">{html.escape(str(profile["summary"]))}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="summary-card">No company summary available.</div>',
                    unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"**Sector:** {html.escape(str(profile.get('sector','N/A') if profile else 'N/A'))}")
    with sc2:
        st.markdown(f"**Industry:** {html.escape(str(profile.get('industry','N/A') if profile else 'N/A'))}")

# ── RIGHT COLUMN: AI Chat ─────────────────────────────────────────────────────
with right:
    st.markdown("""
        <div class="chat-aside">
            <h3>AI assistant</h3>
            <p class="muted">Ask about the business, risks, or how to read the chart — answers use profile + latest price context.</p>
        </div>
    """, unsafe_allow_html=True)

    history_key = f"chat_history_{symbol}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input(f"Ask about {symbol}")
    if user_prompt:
        st.session_state[history_key].append({"role": "user", "content": user_prompt})
        with st.spinner("Thinking..."):
            assistant_reply = get_llm_response(
                symbol=symbol, current_price=current_price,
                current_date=current_date, profile=profile,
                user_prompt=user_prompt,
            )
        st.session_state[history_key].append({"role": "assistant", "content": assistant_reply})
        st.rerun()
