"""
8_Goal_Planner.py

Goal-based financial planner with Monte Carlo simulation.

User enters: target amount, current savings, monthly contribution, horizon (yrs),
and risk profile. We:
  1. Suggest an asset mix (equity / bond / cash) based on risk + horizon.
  2. Simulate N (default 5000) wealth paths using lognormal monthly returns
     parameterized from the asset mix.
  3. Show probability of reaching the goal, percentile bands, required return
     to *just* hit the goal, and the SIP top-up needed.
  4. Optional Groq-powered action plan with concrete next steps.

Pure-numpy implementation; no scipy needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from llm_chat import (
    chat_completion,
    get_gemini_api_key,
    get_gemini_model,
)


st.set_page_config(
    page_title="Goal planner — InvestIQ",
    page_icon="🎯",
    layout="wide",
)


# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg:#080c14; --surface:rgba(17,24,39,0.85);
        --border:rgba(99,202,183,0.18); --accent:#00e5b4; --accent2:#3b82f6;
        --ink:#f0f4f8; --ink-muted:#94a3b8;
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
        max-width:80rem !important; padding:1rem clamp(1rem,3vw,2.5rem) 4rem;
    }
    [data-testid="stSidebar"]{display:none !important;}
    h1 {
        font-family:"Syne",sans-serif; letter-spacing:-0.03em;
        background:linear-gradient(135deg,#f0f4f8 30%,#00e5b4 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .gp-card {
        background:var(--surface); border:1px solid var(--border);
        border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:1rem;
    }
    .gp-strong-pos {color:#22c55e;}
    .gp-strong-neg {color:#ef4444;}

    /* ── Hero (matches Browse stocks style) ── */
    .gp-hero { padding: 1.5rem 0 1rem; }
    .gp-hero .badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em;
        text-transform: uppercase; color: var(--accent);
        background: rgba(0,229,180,0.08);
        border: 1px solid rgba(0,229,180,0.25);
        border-radius: 4px; padding: 0.3rem 0.75rem; margin-bottom: 0.75rem;
    }
    .gp-hero .main-title {
        font-family: "Syne", sans-serif;
        font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 800;
        letter-spacing: -0.04em; line-height: 1.05; margin: 0 0 0.6rem;
        background: linear-gradient(135deg, #f0f4f8 30%, #00e5b4 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .gp-hero .sub-title {
        font-size: clamp(0.9rem, 2vw, 1rem); color: var(--ink-muted);
        max-width: 42rem; line-height: 1.65; margin: 0;
    }
    /* Goal planner §2 KPI row (avoids st.metric truncation / uneven colors) */
    .gp-kpi-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.35rem 0 1rem;
    }
    @media (max-width: 900px) {
        .gp-kpi-row { grid-template-columns: 1fr 1fr; }
    }
    .gp-kpi {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.95rem 1rem;
        min-width: 0;
    }
    .gp-kpi-lbl {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ink-muted);
        margin-bottom: 0.4rem;
        line-height: 1.25;
    }
    .gp-kpi-val {
        font-size: clamp(1rem, 2.1vw, 1.32rem);
        font-weight: 600;
        color: var(--ink);
        line-height: 1.35;
        word-break: break-word;
    }
    .gp-kpi-val .gp-range-lo,
    .gp-kpi-val .gp-range-hi,
    .gp-kpi-val .gp-range-sep { color: var(--ink); font-weight: 600; }
    .gp-kpi-val.gp-ontrack { color: #22c55e; }
    .gp-kpi-val.gp-short { color: #f0f4f8; }
</style>
    """,
    unsafe_allow_html=True,
)

from theme import inject_theme, render_navbar  # noqa: E402
inject_theme()
render_navbar(current="goals")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="gp-hero">
        <div class="badge">● Goal</div>
        <div class="main-title">Goal planner</div>
        <div class="sub-title">Set a financial goal and simulate thousands of market scenarios to see how likely you are to reach it — and what to change if you're behind.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Risk → assumed return distributions (annualized; sourced from long-run
#     historical equity vs bond aggregates as reasonable defaults) ─────────────
RISK_PROFILES: dict[str, dict] = {
    "Conservative": {"equity": 0.30, "bonds": 0.60, "cash": 0.10, "mu": 0.055, "sigma": 0.07},
    "Balanced":     {"equity": 0.55, "bonds": 0.40, "cash": 0.05, "mu": 0.075, "sigma": 0.11},
    "Growth":       {"equity": 0.75, "bonds": 0.20, "cash": 0.05, "mu": 0.090, "sigma": 0.14},
    "Aggressive":   {"equity": 0.90, "bonds": 0.05, "cash": 0.05, "mu": 0.100, "sigma": 0.18},
}


# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown("### 1 · Your goal")

c1, c2, c3 = st.columns(3)
with c1:
    goal_amount = st.number_input(
        "Target amount ($)", min_value=1000.0, max_value=100_000_000.0,
        value=1_000_000.0, step=10_000.0, format="%.0f",
    )
with c2:
    horizon_years = st.slider("Horizon (years)", min_value=1, max_value=40, value=15)
with c3:
    risk_profile = st.selectbox(
        "Risk profile", options=list(RISK_PROFILES.keys()), index=1,
    )

c4, c5, c6 = st.columns(3)
with c4:
    current_savings = st.number_input(
        "Current savings ($)", min_value=0.0, max_value=100_000_000.0,
        value=50_000.0, step=1_000.0, format="%.0f",
    )
with c5:
    monthly_contribution = st.number_input(
        "Monthly contribution ($)", min_value=0.0, max_value=1_000_000.0,
        value=1_000.0, step=100.0, format="%.0f",
    )
with c6:
    n_paths = st.select_slider(
        "Simulation paths",
        options=[1000, 2500, 5000, 10000],
        value=5000,
        help="More paths = smoother probability estimate.",
    )

profile = RISK_PROFILES[risk_profile]
mu_a, sigma_a = profile["mu"], profile["sigma"]


# ── Asset mix card ────────────────────────────────────────────────────────────
mix_left, mix_right = st.columns([1, 1])

with mix_left:
    st.markdown("#### Suggested asset mix")
    mix_df = pd.DataFrame({
        "Asset": ["Equities", "Bonds", "Cash"],
        "Weight": [profile["equity"], profile["bonds"], profile["cash"]],
    })
    fig_mix = go.Figure(
        go.Pie(
            labels=mix_df["Asset"],
            values=mix_df["Weight"],
            hole=0.6,
            marker_colors=["#00e5b4", "#3b82f6", "#94a3b8"],
            textinfo="label+percent",
        )
    )
    fig_mix.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f0f4f8",
        height=260,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_mix, use_container_width=True)

with mix_right:
    st.markdown("#### Assumed long-run stats")
    st.markdown(
        f"""
        <div class="gp-card">
        <p style="margin:0 0 0.5rem;color:var(--ink-muted);font-size:0.8rem;">Profile: <strong>{risk_profile}</strong></p>
        <ul style="margin:0;padding-left:1rem;line-height:1.7;">
          <li>Expected annual return: <strong>{mu_a * 100:.1f}%</strong></li>
          <li>Annual volatility: <strong>{sigma_a * 100:.1f}%</strong></li>
          <li>Equity / Bonds / Cash: <strong>{profile['equity']*100:.0f} / {profile['bonds']*100:.0f} / {profile['cash']*100:.0f}</strong></li>
        </ul>
        <p style="margin-top:0.75rem;color:var(--ink-muted);font-size:0.78rem;">
        Based on long-run historical aggregates. Past performance is not a guarantee — this is for planning, not a forecast.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Monte Carlo simulation (lognormal monthly returns) ────────────────────────
months = int(horizon_years) * 12
mu_m = mu_a / 12.0
sigma_m = sigma_a / np.sqrt(12.0)

rng = np.random.default_rng(seed=42)
# Returns per month per path
shocks = rng.normal(loc=mu_m - 0.5 * sigma_m**2, scale=sigma_m, size=(int(n_paths), months))
gross = np.exp(shocks)  # monthly multiplicative returns

# Iteratively compound: V[t] = (V[t-1] + contribution) * gross[t]
balances = np.empty((int(n_paths), months + 1), dtype=np.float64)
balances[:, 0] = float(current_savings)
contrib = float(monthly_contribution)
for t in range(1, months + 1):
    balances[:, t] = (balances[:, t - 1] + contrib) * gross[:, t - 1]

final_values = balances[:, -1]
prob_reach = float(np.mean(final_values >= goal_amount) * 100.0)
median_final = float(np.percentile(final_values, 50))
p10_final = float(np.percentile(final_values, 10))
p90_final = float(np.percentile(final_values, 90))

# Required CAGR to *exactly* hit goal with given inputs (approximation by
# binary search on a deterministic compound model).
def _terminal_value(rate_annual: float) -> float:
    r = rate_annual / 12.0
    if abs(r) < 1e-9:
        return float(current_savings) + contrib * months
    growth = (1.0 + r) ** months
    return float(current_savings) * growth + contrib * (growth - 1.0) / r


def _required_rate() -> float | None:
    lo, hi = -0.20, 0.50
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _terminal_value(mid) > goal_amount:
            hi = mid
        else:
            lo = mid
    if abs(_terminal_value((lo + hi) / 2.0) - goal_amount) > goal_amount * 0.05:
        return None
    return (lo + hi) / 2.0


required_rate = _required_rate()

# Required additional monthly contribution at the assumed mu_a to hit goal
def _required_contribution(rate_annual: float) -> float:
    r = rate_annual / 12.0
    if abs(r) < 1e-9:
        return max(0.0, (goal_amount - current_savings) / months)
    growth = (1.0 + r) ** months
    needed = goal_amount - current_savings * growth
    return max(0.0, needed * r / (growth - 1.0))


contrib_for_assumed = _required_contribution(mu_a)


# ── Results header ────────────────────────────────────────────────────────────
st.markdown("### 2 · Likelihood of reaching your goal")

gap = float(goal_amount - median_final)
pct_lo = f"${p10_final:,.0f}"
pct_hi = f"${p90_final:,.0f}"
prob_str = f"{prob_reach:.1f}%" if prob_reach < 10 else f"{prob_reach:.0f}%"
if prob_reach > 0 and prob_reach < 0.05:
    prob_str = f"{prob_reach:.2f}%"

if gap <= 0:
    gap_html = (
        '<div class="gp-kpi-val gp-ontrack">At or above goal (median)</div>'
    )
else:
    gap_html = (
        f'<div class="gp-kpi-val gp-short">Short by ${gap:,.0f}</div>'
    )

st.markdown(
    f"""
<div class="gp-kpi-row">
  <div class="gp-kpi">
    <div class="gp-kpi-lbl">Probability of reaching goal</div>
    <div class="gp-kpi-val">{prob_str}</div>
  </div>
  <div class="gp-kpi">
    <div class="gp-kpi-lbl">Median outcome</div>
    <div class="gp-kpi-val">${median_final:,.0f}</div>
  </div>
  <div class="gp-kpi">
    <div class="gp-kpi-lbl">10th–90th percentile</div>
    <div class="gp-kpi-val">
      <span class="gp-range-lo">{pct_lo}</span><span class="gp-range-sep"> – </span><span class="gp-range-hi">{pct_hi}</span>
    </div>
  </div>
  <div class="gp-kpi">
    <div class="gp-kpi-lbl">Median vs goal</div>
    {gap_html}
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Wealth percentile fan chart ───────────────────────────────────────────────
months_axis = np.arange(months + 1)
percentiles = {
    "p10": np.percentile(balances, 10, axis=0),
    "p25": np.percentile(balances, 25, axis=0),
    "p50": np.percentile(balances, 50, axis=0),
    "p75": np.percentile(balances, 75, axis=0),
    "p90": np.percentile(balances, 90, axis=0),
}

fig = go.Figure()

# 10–90 band
fig.add_trace(go.Scatter(
    x=np.concatenate([months_axis, months_axis[::-1]]),
    y=np.concatenate([percentiles["p90"], percentiles["p10"][::-1]]),
    fill="toself", fillcolor="rgba(0,229,180,0.10)",
    line=dict(color="rgba(0,229,180,0.0)"),
    name="10th–90th percentile", hoverinfo="skip",
))
# 25–75 band
fig.add_trace(go.Scatter(
    x=np.concatenate([months_axis, months_axis[::-1]]),
    y=np.concatenate([percentiles["p75"], percentiles["p25"][::-1]]),
    fill="toself", fillcolor="rgba(0,229,180,0.18)",
    line=dict(color="rgba(0,229,180,0.0)"),
    name="25th–75th percentile", hoverinfo="skip",
))
# Median
fig.add_trace(go.Scatter(
    x=months_axis, y=percentiles["p50"], mode="lines",
    line=dict(color="#00e5b4", width=2.5), name="Median",
))
# Goal line
fig.add_hline(
    y=goal_amount, line=dict(color="#f59e0b", dash="dash", width=1.5),
    annotation_text=f"Goal ${goal_amount:,.0f}",
    annotation_position="top left", annotation_font_color="#f59e0b",
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#f0f4f8",
    xaxis=dict(title="Months from now", gridcolor="rgba(99,202,183,0.08)"),
    yaxis=dict(title="Portfolio value", tickprefix="$", tickformat=",.0f", gridcolor="rgba(99,202,183,0.08)"),
    height=420,
    margin=dict(t=20, b=10, l=10, r=10),
    legend=dict(orientation="h", y=1.05, x=0),
)
st.plotly_chart(fig, use_container_width=True)


# ── Distribution of final outcomes ────────────────────────────────────────────
st.markdown("### 3 · Distribution of final outcomes")

hist_fig = go.Figure()
hist_fig.add_trace(go.Histogram(
    x=final_values, nbinsx=60, marker_color="#3b82f6",
    opacity=0.8,
))
hist_fig.add_vline(
    x=goal_amount, line=dict(color="#f59e0b", dash="dash", width=2),
    annotation_text=f"Goal ${goal_amount:,.0f}",
    annotation_position="top right", annotation_font_color="#f59e0b",
)
hist_fig.add_vline(
    x=median_final, line=dict(color="#00e5b4", dash="dot", width=2),
    annotation_text="Median",
    annotation_position="top left", annotation_font_color="#00e5b4",
)
hist_fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#f0f4f8",
    xaxis=dict(title="Ending value", tickprefix="$", tickformat=",.0f", gridcolor="rgba(99,202,183,0.08)"),
    yaxis=dict(title="# scenarios", gridcolor="rgba(99,202,183,0.08)"),
    height=320, bargap=0.02,
    margin=dict(t=10, b=10, l=10, r=10),
)
st.plotly_chart(hist_fig, use_container_width=True)


# ── What needs to change to hit the goal ──────────────────────────────────────
st.markdown("### 4 · What it would take to hit the goal")
st.markdown(
    f"""
    <div class="gp-card">
    <ul style="margin:0;padding-left:1rem;line-height:1.8;">
      <li>Required CAGR (deterministic): <strong>{(required_rate * 100):.1f}% / year</strong>{' — higher than current profile (' + f'{mu_a*100:.1f}%' + ')' if (required_rate is not None and required_rate > mu_a) else ''}</li>
      <li>Monthly contribution needed at the assumed return ({mu_a*100:.1f}%):
          <strong>${contrib_for_assumed:,.0f}/month</strong>
          ({'+' if contrib_for_assumed > monthly_contribution else ''}{contrib_for_assumed - monthly_contribution:+,.0f} vs current)</li>
      <li>Doubling horizon to {horizon_years*2} years would dramatically widen the success band.</li>
    </ul>
    </div>
    """ if required_rate is not None else
    f"""
    <div class="gp-card">
    Current inputs already hit the goal at the assumed return. You may be able to lower contributions or take less risk.
    </div>
    """,
    unsafe_allow_html=True,
)


# ── AI action plan ────────────────────────────────────────────────────────────
st.markdown("### 5 · AI action plan")

plan_col1, plan_col2 = st.columns([3, 1])
with plan_col2:
    plan_btn = st.button("Generate AI plan", type="primary", use_container_width=True)

if plan_btn:
    if not get_gemini_api_key():
        st.warning(
            "GROQ_API_KEY not set. Add it to `.env` (`GROQ_API_KEY=...`) and restart Streamlit."
        )
    else:
        cagr_line = (
            f"- Required CAGR: {required_rate * 100:.1f}%\n"
            if required_rate is not None else ""
        )
        prompt = (
            "You are a financial planning coach. Be concise (under 250 words). "
            "Use only the numbers provided. Do not recommend specific securities.\n\n"
            f"Goal: ${goal_amount:,.0f} in {horizon_years} years\n"
            f"Current savings: ${current_savings:,.0f}\n"
            f"Monthly contribution: ${monthly_contribution:,.0f}\n"
            f"Risk profile: {risk_profile} "
            f"(equity {profile['equity'] * 100:.0f}%, bonds {profile['bonds'] * 100:.0f}%, "
            f"cash {profile['cash'] * 100:.0f}%; "
            f"assumed return {mu_a * 100:.1f}% / vol {sigma_a * 100:.1f}%)\n\n"
            "Monte Carlo result:\n"
            f"- Probability of reaching goal: {prob_str}\n"
            f"- Median final value: ${median_final:,.0f}\n"
            f"- 10th–90th percentile: ${p10_final:,.0f} – ${p90_final:,.0f}\n"
            + cagr_line +
            f"- Required monthly contribution at current return: ${contrib_for_assumed:,.0f}\n\n"
            "Output exactly four sections in markdown:\n"
            "**Where you stand** — 2 bullets describing the current trajectory.\n"
            "**Top 3 levers to improve odds** — each lever with rough impact "
            "(e.g. +1% return, +$200/mo).\n"
            "**Behavioral risks to watch** — 2 bullets (sequence-of-returns, lifestyle creep, etc.).\n"
            "**12-month checklist** — 3–5 concrete actions.\n"
            
        )
        with st.spinner("Generating your plan…"):
            try:
                model = get_gemini_model()
                plan = chat_completion(
                    [{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.3,
                )
                st.markdown(
                    f"""<div class="gp-card">{plan}</div>""",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"AI plan failed: {e}")


