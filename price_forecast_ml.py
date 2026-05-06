"""
price_forecast_ml.py
====================
Three models in one file:
  1. ARIMA         – price forecast with confidence band (original)
  2. RF Classifier – Buy / Hold / Avoid signal (original)
  3. RF Regressor  – actual price prediction 5/10/20 days
  4. Sentiment     – FinBERT with VADER fallback

Educational use only — not investment advice.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None  # type: ignore[misc, assignment]

from pandas.tseries.offsets import BDay


# ══════════════════════════════════════════════════════════════
# ARIMA FORECAST  (original — with linear trend fallback added)
# ══════════════════════════════════════════════════════════════

@dataclass
class ForecastResult:
    forecast_dates: list[pd.Timestamp]
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    model_order: tuple
    n_obs: int


def _linear_trend_forecast(
    price_df: pd.DataFrame,
    horizon: int,
    confidence: float,
) -> ForecastResult | None:
    """
    Pure numpy fallback when ARIMA fails to converge.

    Why ARIMA fails on some stocks (e.g. NKE):
    - Dramatic price crashes cause non-constant variance
    - The optimizer cannot find stable parameters across both
      the "normal" and "crashed" price periods
    - All 4 ARIMA orders fail → returns None → error shown

    This fallback fits a log-linear trend on the last 180 days
    and projects forward. Confidence band widens as sqrt(t)
    using historical daily return std. Works on any price series.
    """
    df = price_df.sort_values("report_date").dropna(subset=["close"])
    closes = df["close"].astype(float).values
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

    # z-score for confidence level
    z = 1.96 if abs(confidence - 0.95) < 0.01 else (
        1.645 if abs(confidence - 0.90) < 0.01 else 2.576
    )

    last_dt = pd.Timestamp(df["report_date"].iloc[-1])
    future_dates: list[pd.Timestamp] = []
    d = last_dt
    for _ in range(horizon):
        d = d + BDay(1)
        future_dates.append(d)

    t = np.arange(1, horizon + 1, dtype=float)
    log_point = intercept + slope * (window - 1 + t)
    point = np.exp(log_point)
    band  = z * sigma * np.sqrt(t)
    lower = np.exp(log_point - band)
    upper = np.exp(log_point + band)

    return ForecastResult(
        forecast_dates=future_dates,
        point=point,
        lower=lower,
        upper=upper,
        model_order=("linear trend",),
        n_obs=int(len(closes)),
    )


def arima_forecast_confidence(
    price_df: pd.DataFrame,
    *,
    horizon: int = 30,
    confidence: float = 0.95,
    min_obs: int = 50,
) -> ForecastResult | None:
    if price_df.empty or len(price_df) < min_obs:
        return None

    # If statsmodels not installed, use linear trend directly
    if ARIMA is None:
        return _linear_trend_forecast(price_df, horizon, confidence)

    df = price_df.sort_values("report_date").dropna(subset=["close"])
    y = df["close"].astype(float).values
    if len(y) < min_obs or not np.all(np.isfinite(y)):
        return None

    alpha = 1.0 - confidence
    orders: list[tuple[int, int, int]] = [(2, 1, 2), (1, 1, 1), (1, 1, 0), (0, 1, 1)]
    res_fit = None
    used_order = (1, 1, 1)
    for order in orders:
        try:
            res_fit = ARIMA(y, order=order).fit()
            used_order = order
            break
        except Exception:
            continue

    # If all ARIMA orders fail to converge, use linear trend fallback
    if res_fit is None:
        return _linear_trend_forecast(price_df, horizon, confidence)

    try:
        fc = res_fit.get_forecast(steps=horizon)
        pred = np.asarray(fc.predicted_mean, dtype=float)
        conf_int = fc.conf_int(alpha=alpha)
        lower = np.asarray(conf_int.iloc[:, 0], dtype=float)
        upper = np.asarray(conf_int.iloc[:, 1], dtype=float)
    except Exception:
        return _linear_trend_forecast(price_df, horizon, confidence)

    last_dt = pd.Timestamp(df["report_date"].iloc[-1])
    future_dates: list[pd.Timestamp] = []
    d = last_dt
    for _ in range(horizon):
        d = d + BDay(1)
        future_dates.append(d)

    return ForecastResult(
        forecast_dates=future_dates, point=pred, lower=lower, upper=upper,
        model_order=used_order, n_obs=len(y),
    )


def build_forecast_chart_figure(hist_df: pd.DataFrame, fc: ForecastResult, symbol: str):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["report_date"], y=hist_df["close"], name="History",
        line=dict(color="#5f3a73", width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Close: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fc.forecast_dates, y=fc.lower, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fc.forecast_dates, y=fc.upper, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(196, 74, 122, 0.22)", name="~95% interval",
        hovertemplate="Upper: $%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fc.forecast_dates, y=fc.point, name="Forecast",
        line=dict(color="#c44a7a", width=2, dash="dash"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: $%{y:.2f}<extra></extra>",
    ))
    # Show model type in title
    model_label = (
        f"ARIMA{fc.model_order}"
        if fc.model_order != ("linear trend",)
        else "Linear trend (ARIMA fallback)"
    )
    fig.update_layout(
        title=f"{symbol} — {model_label} (n={fc.n_obs})",
        template="plotly_white", height=440,
        paper_bgcolor="rgba(255,255,255,0.5)", plot_bgcolor="rgba(255,255,255,0.35)",
        font=dict(color="#1e1428"), margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="rgba(90,70,100,0.12)"),
        yaxis=dict(title="Price (USD)", gridcolor="rgba(90,70,100,0.12)"),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# SHARED FEATURE ENGINEERING  (used by both RF models)
# ══════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "mom_5d", "mom_20d", "mom_60d",
    "price_vs_sma20", "price_vs_sma50", "sma_cross",
    "volatility_20d", "volatility_60d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "vol_ratio", "pos_52w",
]

LABEL_NAMES = {0: "Avoid", 1: "Hold", 2: "Buy"}
CONFIDENCE_THRESHOLD = 0.55


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_rf_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("report_date").reset_index(drop=True)
    c = df["close"]
    df["mom_5d"]  = c.pct_change(5)
    df["mom_20d"] = c.pct_change(20)
    df["mom_60d"] = c.pct_change(60)
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    df["sma_cross"]      = (sma20 - sma50) / (sma50 + 1e-9)
    df["price_vs_sma20"] = (c - sma20) / (sma20 + 1e-9)
    df["price_vs_sma50"] = (c - sma50) / (sma50 + 1e-9)
    df["volatility_20d"] = c.pct_change().rolling(20).std()
    df["volatility_60d"] = c.pct_change().rolling(60).std()
    df["rsi_14"] = _compute_rsi(c, 14)
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
    df["vol_ratio"] = vol / (vol.rolling(20).mean() + 1e-9)
    high52 = c.rolling(252).max()
    low52  = c.rolling(252).min()
    df["pos_52w"] = (c - low52) / (high52 - low52 + 1e-9)
    df["fwd_return"] = c.shift(-20) / (c + 1e-9) - 1
    return df


def _label_signal(r: float) -> int:
    if r >  0.04: return 2
    if r < -0.02: return 0
    return 1


# ══════════════════════════════════════════════════════════════
# RF CLASSIFIER — Buy / Hold / Avoid / Abstain
#
# Abstain rule (your logic):
# When the top two signal probabilities are within 3% of each
# other the model is genuinely tied — forcing a decision would
# be misleading. Abstain is shown instead.
#
# Example:
#   Buy 37%, Avoid 37%, Hold 26% → tied → Abstain ✅
#   Hold 43%, Buy 32%, Avoid 25% → clear → Hold ✅
# ══════════════════════════════════════════════════════════════

def run_rf_signal(price_df: pd.DataFrame) -> dict | None:
    """
    Random Forest CLASSIFIER → Buy / Hold / Avoid / Abstain.

    Abstains when the top two probabilities are within 3% —
    a genuine tie means the model cannot distinguish between them.

    Risk score uses avg of 20d+60d volatility to align with the
    homepage stock_metrics 90-day definition.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.inspection import permutation_importance
    except ImportError:
        return None

    if price_df.empty or len(price_df) < 150:
        return None

    df = _compute_rf_features(price_df)
    df["label"] = df["fwd_return"].apply(
        lambda x: _label_signal(x) if pd.notna(x) else np.nan
    )
    df = df.dropna(subset=FEATURE_COLS + ["label"])
    if len(df) < 100:
        return None

    df["label"] = df["label"].astype(int)
    X = df[FEATURE_COLS].values
    y = df["label"].values

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=5,
            class_weight="balanced", random_state=42,
        )
        clf.fit(X[train_idx], y[train_idx])
        cv_scores.append(clf.score(X[val_idx], y[val_idx]))

    split = int(len(X) * 0.8)
    base_clf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42,
    )
    base_clf.fit(X[:split], y[:split])
    cal_clf = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")
    cal_clf.fit(X, y)

    perm = permutation_importance(
        base_clf, X[split:], y[split:],
        n_repeats=10, random_state=42, scoring="f1_macro",
    )
    feat_imp = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=False)

    last_X     = df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
    proba      = cal_clf.predict_proba(last_X)[0]
    classes    = cal_clf.classes_
    prob_dict  = {LABEL_NAMES[c]: round(float(proba[i]), 3) for i, c in enumerate(classes)}
    max_prob   = float(max(proba))
    pred_class = classes[int(np.argmax(proba))]
    raw_signal = LABEL_NAMES[pred_class]

    # ── Abstain when top two probabilities are within 3% (tied) ──────────
    sorted_probs = sorted(proba, reverse=True)
    top_gap = float(sorted_probs[0] - sorted_probs[1])
    low_confidence = top_gap <= 0.03
    final = "Abstain" if low_confidence else raw_signal

    latest     = df.iloc[-1]
    vol_val    = float(latest["volatility_20d"]) if pd.notna(latest["volatility_20d"]) else 0.02
    vol_60_val = float(latest["volatility_60d"]) if pd.notna(latest["volatility_60d"]) else vol_val
    rsi_val    = float(latest["rsi_14"])          if pd.notna(latest["rsi_14"])          else 50
    pos_val    = float(latest["pos_52w"])          if pd.notna(latest["pos_52w"])          else 0.5

    # Risk score aligned with homepage 90-day volatility definition
    avg_vol   = (vol_val + vol_60_val) / 2
    vol_score = min(8.0, avg_vol * 250)
    rsi_score = abs(rsi_val - 50) / 50 * 1.5
    pos_score = (1 - pos_val) * 1.0
    risk_score = round(min(10.0, max(1.0, vol_score + rsi_score + pos_score)), 1)
    risk_label = "Low" if risk_score <= 3.5 else ("Medium" if risk_score <= 6.5 else "High")

    return {
        "signal":         final,
        "raw_signal":     raw_signal,
        "confidence":     round(max_prob, 3),
        "probabilities":  prob_dict,
        "risk_score":     risk_score,
        "risk_label":     risk_label,
        "low_confidence": low_confidence,   # True when top two probs within 3%
        "top_gap":        round(top_gap, 3),
        "cv_accuracy":    round(float(np.mean(cv_scores)), 3),
        "cv_std":         round(float(np.std(cv_scores)),  3),
        "n_training":     len(df),
        "feat_imp":       feat_imp,
        "latest": {
            "RSI (14)":       round(rsi_val, 1),
            "Momentum 20d":   f"{latest['mom_20d']*100:.1f}%",
            "Momentum 60d":   f"{latest['mom_60d']*100:.1f}%",
            "vs SMA-20":      f"{latest['price_vs_sma20']*100:.1f}%",
            "vs SMA-50":      f"{latest['price_vs_sma50']*100:.1f}%",
            "Volatility 20d": f"{latest['volatility_20d']*100:.2f}%",
            "52w Position":   f"{latest['pos_52w']*100:.0f}%",
        },
    }


# ══════════════════════════════════════════════════════════════
# ITEM 4 — RF REGRESSOR  (unchanged)
# ══════════════════════════════════════════════════════════════

def run_rf_regression(price_df: pd.DataFrame) -> dict | None:
    """RF Regression: predict price in 5, 10, 20 trading days."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError:
        return None

    if price_df.empty or len(price_df) < 150:
        return None

    df = _compute_rf_features(price_df)
    current_price = float(df["close"].iloc[-1])
    results: dict = {}

    for horizon in [5, 10, 20]:
        col = f"fwd_{horizon}d"
        df[col] = df["close"].shift(-horizon) / (df["close"] + 1e-9) - 1

        train_df = df.dropna(subset=FEATURE_COLS + [col])
        if len(train_df) < 100:
            continue

        X = train_df[FEATURE_COLS].values
        y = train_df[col].values

        tscv = TimeSeriesSplit(n_splits=5)
        mape_scores = []
        for train_idx, val_idx in tscv.split(X):
            reg = RandomForestRegressor(
                n_estimators=100, max_depth=6,
                min_samples_leaf=5, random_state=42,
            )
            reg.fit(X[train_idx], y[train_idx])
            pred   = reg.predict(X[val_idx])
            actual = y[val_idx]
            mask   = np.abs(actual) > 0.001
            if mask.sum() > 0:
                mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
                mape_scores.append(float(mape))

        reg_final = RandomForestRegressor(
            n_estimators=200, max_depth=6,
            min_samples_leaf=5, random_state=42,
        )
        reg_final.fit(X, y)

        last_X = train_df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
        tree_preds  = np.array([tree.predict(last_X)[0] for tree in reg_final.estimators_])
        pred_return = float(np.mean(tree_preds))
        pred_std    = float(np.std(tree_preds))

        pred_price  = current_price * (1 + pred_return)
        low_price   = current_price * (1 + pred_return - 1.96 * pred_std)
        high_price  = current_price * (1 + pred_return + 1.96 * pred_std)

        results[horizon] = {
            "predicted_price":  round(pred_price, 2),
            "low":              round(max(0.0, low_price), 2),
            "high":             round(high_price, 2),
            "predicted_return": round(pred_return * 100, 2),
            "mape":             round(float(np.mean(mape_scores)), 2) if mape_scores else None,
        }

    if not results:
        return None

    return {
        "current_price": round(current_price, 2),
        "predictions":   results,
    }


# ══════════════════════════════════════════════════════════════
# ITEM 2 — NEWS SENTIMENT  (unchanged)
# ══════════════════════════════════════════════════════════════

def get_sentiment_score(symbol: str) -> dict:
    """Fetch recent Yahoo Finance headlines and score sentiment.
    Primary: FinBERT. Fallback: VADER."""
    try:
        import yfinance as yf
        headlines = _fetch_headlines(symbol, yf)
        if not headlines:
            return {"score": 0.0, "label": "Neutral", "count": 0}
        result = _sentiment_via_finbert(headlines)
        if result:
            return result
        return _sentiment_via_vader(headlines)
    except Exception as exc:
        return {"score": 0.0, "label": "Neutral", "count": 0, "error": str(exc)}


def _fetch_headlines(symbol: str, yf) -> list[str]:
    ticker = yf.Ticker(symbol.strip().upper())
    news   = ticker.news or []
    texts: list[str] = []
    for item in news[:20]:
        if not isinstance(item, dict):
            continue
        c = item.get("content", item)
        if isinstance(c, dict):
            title   = str(c.get("title")   or "")
            summary = str(c.get("summary")  or "")
        else:
            title   = str(item.get("title")   or "")
            summary = str(item.get("summary")  or "")
        text = f"{title}. {summary}".strip(". ")[:400]
        if text:
            texts.append(text)
    return texts


def _sentiment_via_finbert(headlines: list[str]) -> dict | None:
    try:
        from transformers import pipeline
        finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            device=-1,
            truncation=True,
            max_length=512,
        )
        results = finbert(headlines)

        def _to_score(result_list: list[dict]) -> float:
            mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            return sum(
                mapping.get(r["label"].lower(), 0.0) * float(r["score"])
                for r in result_list
            )

        scores = [_to_score(r) for r in results]
        avg   = float(sum(scores) / len(scores))
        label = "Positive" if avg >= 0.1 else ("Negative" if avg <= -0.1 else "Neutral")
        return _build_sentiment_result(
            scores, avg, label,
            summary=f"FinBERT analysis of {len(scores)} financial headlines.",
            model="ProsusAI/finbert (RoBERTa fine-tuned on financial text)",
        )
    except ImportError:
        return None
    except Exception:
        return None


def _sentiment_via_vader(headlines: list[str]) -> dict:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores   = [analyzer.polarity_scores(h)["compound"] for h in headlines if h]
        if not scores:
            return {"score": 0.0, "label": "Neutral", "count": 0, "model": "VADER (fallback)"}
        avg   = float(np.mean(scores))
        label = "Positive" if avg >= 0.05 else ("Negative" if avg <= -0.05 else "Neutral")
        return _build_sentiment_result(
            scores, avg, label,
            summary="VADER word-level sentiment analysis.",
            model="VADER (fallback — FinBERT unavailable)",
        )
    except Exception:
        return {"score": 0.0, "label": "Neutral", "count": 0, "model": "unavailable"}


def _build_sentiment_result(scores: list[float], avg: float, label: str,
                             summary: str, model: str) -> dict:
    pos_pct = round(sum(1 for s in scores if s >= 0.05)  / len(scores) * 100)
    neg_pct = round(sum(1 for s in scores if s <= -0.05) / len(scores) * 100)
    neu_pct = max(0, 100 - pos_pct - neg_pct)
    recent  = scores[:7]
    older   = scores[7:]
    r_avg   = float(np.mean(recent)) if recent else avg
    o_avg   = float(np.mean(older))  if older  else avg
    shift   = r_avg - o_avg
    if shift > 0.12:
        shift_label = "Improving ↑"
    elif shift < -0.12:
        shift_label = "Deteriorating ↓"
    else:
        shift_label = "Stable →"
    return {
        "score":        round(avg, 3),
        "label":        label,
        "count":        len(scores),
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "neutral_pct":  neu_pct,
        "recent_avg":   round(r_avg, 3),
        "older_avg":    round(o_avg, 3),
        "shift":        round(shift, 3),
        "shift_label":  shift_label,
        "summary":      summary,
        "model":        model,
    }
