"""
3_Compare_Stocks.py
Side-by-side stock comparison.
Item 7: ML signals + sentiment added after existing comparison table.

Fix: session state used so clicking "Run ML comparison" doesn't
lose the comparison table (was refreshing to empty page before).
"""

from pathlib import Path
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from investiq_data import (
    company_label_options,
    load_homepage_data,
    load_price_snapshot,
    load_stock_metrics,
    snapshot_for_symbols,
    yahoo_style_comparison_table,
)

st.set_page_config(
    page_title="Compare stocks — InvestIQ",
    page_icon="⚖️",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
# This is the key fix: store comparison results so they survive button reruns.
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None
if "cmp_snap" not in st.session_state:
    st.session_state.cmp_snap = None
if "cmp_picked" not in st.session_state:
    st.session_state.cmp_picked = []
if "cmp_ml_results" not in st.session_state:
    st.session_state.cmp_ml_results = None



def _load_data_with_error_handling():
    """Load data with generic error handling."""
    try:
        df = load_homepage_data()
        latest_prices = load_price_snapshot()
        with st.spinner("Loading metrics…"):
            stock_metrics = load_stock_metrics()
        return df, latest_prices, stock_metrics
    except Exception as exc:
        st.error(f"Error loading data: {exc}")
        st.info("Please ensure you have a stable internet connection for yfinance data fetching.")
        st.stop()


st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg:#080c14;--bg2:#0d1220;--bg3:#111827;
        --surface:rgba(17,24,39,0.85);--surface2:rgba(30,41,59,0.7);
        --border:rgba(99,202,183,0.12);--border2:rgba(99,202,183,0.22);
        --accent:#00e5b4;--accent2:#3b82f6;
        --ink:#f0f4f8;--ink-muted:#94a3b8;--ink-dim:#475569;
        --shadow:0 20px 60px rgba(0,0,0,0.5);--glow:0 0 30px rgba(0,229,180,0.12);
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
    .cmp-hero{padding:1.5rem 0 0.75rem;}
    .cmp-badge{
        display:inline-flex;align-items:center;gap:0.4rem;
        font-family:"IBM Plex Mono",monospace;font-size:0.68rem;font-weight:600;
        letter-spacing:0.15em;text-transform:uppercase;color:var(--accent);
        background:rgba(0,229,180,0.08);border:1px solid rgba(0,229,180,0.25);
        border-radius:4px;padding:0.3rem 0.75rem;margin-bottom:0.75rem;
    }
    .cmp-title{
        font-family:"Syne",sans-serif;font-size:clamp(2rem,5vw,3rem);font-weight:800;
        letter-spacing:-0.04em;margin:0 0 0.5rem;
        background:linear-gradient(135deg,#f0f4f8 30%,#00e5b4 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    }
    .cmp-sub{font-size:0.95rem;color:var(--ink-muted);max-width:40rem;line-height:1.6;}
    div.stButton > button[kind="primary"]{
        border-radius:8px !important;border:none !important;
        background:linear-gradient(135deg,#00e5b4 0%,#00c49a 100%) !important;
        color:#080c14 !important;font-weight:700 !important;font-family:"Syne",sans-serif !important;
        box-shadow:0 8px 30px rgba(0,229,180,0.2) !important;
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
    .stMultiSelect > div > div{
        background:var(--bg3) !important;border:1px solid var(--border) !important;
        border-radius:8px !important;color:var(--ink) !important;
    }
    .block-container label,.stMultiSelect label{
        color:var(--ink-muted) !important;font-size:0.8rem !important;
        font-weight:500 !important;letter-spacing:0.03em !important;text-transform:uppercase !important;
    }
    [data-testid="stDataFrame"]{border-radius:10px !important;border:1px solid var(--border) !important;}
    [data-testid="stAlert"]{
        border-radius:8px !important;border:1px solid var(--border) !important;background:var(--bg3) !important;
    }
    hr{border:none !important;border-top:1px solid var(--border) !important;}
</style>
""", unsafe_allow_html=True)

from theme import inject_theme, render_navbar  # noqa: E402
inject_theme()
render_navbar(current="compare")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="cmp-hero">
        <div class="cmp-badge">● Compare · Side-by-side</div>
        <div class="cmp-title">Compare stocks</div>
        <div class="cmp-sub">Pick 2–5 tickers. Metrics are <strong>rows</strong> and symbols are <strong>columns</strong>.</div>
    </div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df, latest_prices, stock_metrics = _load_data_with_error_handling()

compare_options, label_to_symbol = company_label_options(df)

st.markdown("---")
picked_labels = st.multiselect(
    "Companies to compare",
    options=compare_options,
    default=[],
    max_selections=5,
    help="Full company names with tickers. Choose 2–5.",
)

run = st.button("Run comparison", type="primary")

# ── When "Run comparison" clicked → save to session state ─────────────────────
if run:
    picked = [label_to_symbol[lab] for lab in picked_labels]
    if len(picked) < 2:
        st.warning("Choose at least two companies.")
    elif df.empty or latest_prices.empty:
        st.error("No market data loaded.")
    else:
        snap = snapshot_for_symbols(picked, df, latest_prices, stock_metrics)
        if snap.empty:
            st.warning("No rows returned for those symbols — check spelling and dataset coverage.")
        else:
            # Save to session state so it survives subsequent button clicks
            st.session_state.cmp_snap   = snap
            st.session_state.cmp_picked = picked
            st.session_state.cmp_ml_results = None  # reset ML results for new comparison

# ── Display comparison (from session state — survives reruns) ─────────────────
if st.session_state.cmp_snap is not None and not st.session_state.cmp_snap.empty:
    snap   = st.session_state.cmp_snap
    picked = st.session_state.cmp_picked

    # ── EXISTING: fundamentals table (unchanged) ──────────────────────────────
    cmp_df = yahoo_style_comparison_table(snap)
    st.caption("Rows = fundamentals from lakehouse. Open Details per ticker for charts and AI Q&A.")
    st.dataframe(cmp_df, use_container_width=True, height=min(520, 56*(len(cmp_df)+2)))

    st.markdown("##### Open company page")
    bt = st.columns(len(snap))
    for i, (_, row) in enumerate(snap.iterrows()):
        sym = row["symbol"]
        with bt[i]:
            if st.button(f"Details · {sym}", key=f"cmp_det_{sym}", use_container_width=True):
                st.session_state.selected_symbol = sym
                st.switch_page("pages/1_Company_Details.py")

    # ── ITEM 7: ML signals + sentiment (NEW) ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 ML signals & sentiment comparison")
    st.caption(
        "RF Buy/Hold/Avoid signal + news sentiment per company. "
        "Price data fetched live from Yahoo Finance for ML. "
        "Each model trained independently on that company's history."
    )

    run_ml_cmp = st.button(
        "Run ML comparison (~30s per company)",
        type="primary",
        key="run_ml_cmp_btn"
    )

    # When clicked → run ML and save results to session state
    if run_ml_cmp:
        import yfinance as yf
        import pandas as pd
        from price_forecast_ml import run_rf_signal, get_sentiment_score

        ml_results = {}
        progress = st.progress(0, text="Starting...")

        for i, sym in enumerate(picked):
            progress.progress(
                i / len(picked),
                text=f"Analyzing {sym}... ({i+1}/{len(picked)})"
            )
            try:
                ticker = yf.Ticker(sym)
                hist   = ticker.history(period="5y", interval="1d")

                if hist.empty:
                    ml_results[sym] = {"error": "No price data from Yahoo Finance"}
                    continue

                hist = hist.reset_index()
                hist.columns = [c.lower() for c in hist.columns]
                hist = hist.rename(columns={"date": "report_date"})
                hist["report_date"] = pd.to_datetime(hist["report_date"]).dt.tz_localize(None)

                rf   = run_rf_signal(hist)
                sent = get_sentiment_score(sym)

                ml_results[sym] = {"rf": rf, "sent": sent}

            except Exception as e:
                ml_results[sym] = {"error": str(e)}

        progress.progress(1.0, text="Done!")
        st.session_state.cmp_ml_results = ml_results  # save to session state

    # Display ML results from session state (persists across reruns)
    if st.session_state.cmp_ml_results is not None:
        ml_results = st.session_state.cmp_ml_results
        import pandas as pd

        # ── Signal cards ─────────────────────────────────────────────────────
        st.markdown("#### Investment signals")
        sig_cols = st.columns(len(picked))

        for i, sym in enumerate(picked):
            with sig_cols[i]:
                res = ml_results.get(sym, {})

                if "error" in res:
                    st.markdown(f"""
                    <div style="padding:14px;background:rgba(0,0,0,0.2);border-radius:10px;
                                border:1px solid rgba(99,202,183,0.15);text-align:center;">
                        <div style="font-family:'IBM Plex Mono';font-size:0.9rem;
                                    color:#f0f4f8;font-weight:700;">{sym}</div>
                        <div style="color:#ef4444;font-size:12px;margin-top:6px;">
                            {res['error']}
                        </div>
                    </div>""", unsafe_allow_html=True)
                    continue

                rf = res.get("rf")
                if rf is None:
                    st.markdown(f"""
                    <div style="padding:14px;background:rgba(0,0,0,0.2);border-radius:10px;
                                border:1px solid rgba(99,202,183,0.15);text-align:center;">
                        <div style="font-family:'IBM Plex Mono';color:#f0f4f8;font-weight:700;">{sym}</div>
                        <div style="color:#94a3b8;font-size:12px;margin-top:6px;">Insufficient data</div>
                    </div>""", unsafe_allow_html=True)
                    continue

                signal    = rf["signal"]
                sig_emoji = {"Buy":"📈","Hold":"⏸️","Avoid":"📉","Abstain":"🤔"}.get(signal,"🤔")
                sig_color = {"Buy":"#00e564","Hold":"#f59e0b","Avoid":"#ef4444","Abstain":"#94a3b8"}.get(signal,"#94a3b8")

                st.markdown(f"""
                <div style="padding:16px;background:rgba(0,0,0,0.25);border-radius:12px;
                            border:1px solid rgba(99,202,183,0.2);text-align:center;margin-bottom:8px;">
                    <div style="font-family:'IBM Plex Mono';font-size:1rem;
                                color:#f0f4f8;font-weight:700;margin-bottom:8px;">{sym}</div>
                    <div style="font-size:1.8rem;margin-bottom:4px;">{sig_emoji}</div>
                    <div style="font-family:'Syne';font-size:1.2rem;font-weight:800;
                                color:{sig_color};margin-bottom:8px;">{signal}</div>
                    <div style="font-size:12px;color:#94a3b8;">
                        Confidence: <strong style="color:#f0f4f8;">{rf['confidence']*100:.0f}%</strong>
                    </div>
                    <div style="font-size:12px;color:#94a3b8;">
                        Risk: <strong style="color:#f0f4f8;">{rf['risk_score']}/10</strong>
                    </div>
                    <div style="font-size:12px;color:#94a3b8;">
                        CV accuracy: <strong style="color:#f0f4f8;">{rf['cv_accuracy']*100:.1f}%</strong>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Probability bars
                prob_colors = {"Buy":"#00e564","Hold":"#f59e0b","Avoid":"#ef4444"}
                for lbl in ["Buy","Hold","Avoid"]:
                    p = rf["probabilities"].get(lbl, 0)
                    st.markdown(f"""
                    <div style="margin-bottom:5px;">
                      <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px;">
                        <span style="color:#94a3b8;">{lbl}</span>
                        <span style="color:{prob_colors[lbl]};font-weight:600;">{p*100:.0f}%</span>
                      </div>
                      <div style="background:rgba(255,255,255,0.07);border-radius:3px;height:5px;overflow:hidden;">
                        <div style="width:{p*100}%;background:{prob_colors[lbl]};height:100%;border-radius:3px;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

        # ── Sentiment cards ───────────────────────────────────────────────────
        st.markdown("#### News sentiment")
        sent_cols = st.columns(len(picked))

        for i, sym in enumerate(picked):
            with sent_cols[i]:
                res  = ml_results.get(sym, {})
                sent = res.get("sent", {})

                if not sent or sent.get("count", 0) == 0:
                    st.markdown(f"""
                    <div style="padding:14px;background:rgba(0,0,0,0.2);border-radius:10px;
                                border:1px solid rgba(99,202,183,0.15);text-align:center;">
                        <div style="font-family:'IBM Plex Mono';color:#f0f4f8;font-weight:700;">{sym}</div>
                        <div style="color:#94a3b8;font-size:12px;margin-top:6px;">No sentiment data</div>
                    </div>""", unsafe_allow_html=True)
                    continue

                s_color = {
                    "Positive":"#00e564","Negative":"#ef4444","Neutral":"#f59e0b",
                }.get(sent["label"], "#94a3b8")

                shift_color = {
                    "Improving ↑":"#00e564","Deteriorating ↓":"#ef4444","Stable →":"#f59e0b",
                }.get(sent.get("shift_label","Stable →"), "#f59e0b")

                st.markdown(f"""
                <div style="padding:16px;background:rgba(0,0,0,0.25);border-radius:12px;
                            border:1px solid rgba(99,202,183,0.2);text-align:center;margin-bottom:8px;">
                    <div style="font-family:'IBM Plex Mono';font-size:1rem;
                                color:#f0f4f8;font-weight:700;margin-bottom:8px;">{sym}</div>
                    <div style="font-family:'Syne';font-size:1.3rem;font-weight:800;
                                color:{s_color};margin-bottom:4px;">{sent['label']}</div>
                    <div style="font-family:'IBM Plex Mono';font-size:1rem;
                                color:#f0f4f8;margin-bottom:8px;">{sent['score']:+.3f}</div>
                    <div style="font-size:13px;font-weight:600;color:{shift_color};margin-bottom:8px;">
                        {sent.get('shift_label','—')}
                    </div>
                    <div style="font-size:11px;color:#94a3b8;">📰 {sent['count']} headlines scored</div>
                    <div style="display:flex;justify-content:space-around;margin-top:10px;font-size:11px;">
                        <span style="color:#00e564;">▲ {sent['positive_pct']}%</span>
                        <span style="color:#f59e0b;">— {sent['neutral_pct']}%</span>
                        <span style="color:#ef4444;">▼ {sent['negative_pct']}%</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── Summary table ─────────────────────────────────────────────────────
        st.markdown("#### ML comparison summary")
        rows = []
        for sym in picked:
            res  = ml_results.get(sym, {})
            rf   = res.get("rf")
            sent = res.get("sent", {})
            rows.append({
                "Symbol":       sym,
                "ML Signal":    rf["signal"] if rf else "—",
                "Confidence":   f"{rf['confidence']*100:.0f}%" if rf else "—",
                "Risk (1-10)":  str(rf["risk_score"]) if rf else "—",
                "CV Accuracy":  f"{rf['cv_accuracy']*100:.1f}%" if rf else "—",
                "Sentiment":    sent.get("label", "—"),
                "Sent. Score":  f"{sent['score']:+.3f}" if sent.get("count",0) > 0 else "—",
                "Sent. Trend":  sent.get("shift_label", "—"),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("⚠️ ML signals use historical price data only. Sentiment from Yahoo Finance news. Not investment advice.")
