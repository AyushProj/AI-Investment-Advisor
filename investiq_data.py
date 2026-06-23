"""
InvestIQ data layer – Streamlit + yfinance.

Single-phase design
-------------------
All data is fetched directly from yfinance on demand and cached with 
Streamlit's @st.cache_data decorator. No Databricks or external databases 
required. Perfect for direct Streamlit hosting.

Features
--------
  - Direct yfinance integration for real-time market data
  - Streamlit caching for performance (300-900s TTL)
  - Profile, prices, dividends, EPS, and financial statements
  - Works offline with cached data from last fetch
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_TTL_SHORT = 300    # 5 minutes
CACHE_TTL_MEDIUM = 600   # 10 minutes
CACHE_TTL_LONG = 900     # 15 minutes

# Default ticker universe (S&P-100 sample)
DEFAULT_TICKERS: list[str] = [
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


# ---------------------------------------------------------------------------
# Data Fetching – Direct yfinance API
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SHORT)
def _fetch_ticker_info(symbol: str) -> dict:
    """Fetch ticker info from yfinance with error handling."""
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info or {}
    except Exception as exc:
        st.warning(f"Could not fetch info for {symbol}: {exc}")
        return {}


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _fetch_price_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    """Fetch historical OHLCV data from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        hist.index = pd.to_datetime(hist.index).normalize()
        hist = hist.reset_index()
        hist.columns = [c.lower() for c in hist.columns]
        hist["symbol"] = symbol.upper()
        hist = hist.rename(columns={"date": "report_date"})
        return hist
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _fetch_dividends(symbol: str) -> pd.DataFrame:
    """Fetch dividend history from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        divs = ticker.dividends
        if divs is None or divs.empty:
            return pd.DataFrame()
        df = pd.DataFrame({
            "symbol": symbol.upper(),
            "report_date": pd.to_datetime(divs.index).normalize(),
            "amount": divs.values,
        })
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL_MEDIUM)
def _fetch_batch_tickers(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Fetch basic info for multiple tickers at once."""
    records = []
    for symbol in symbols:
        info = _fetch_ticker_info(symbol)
        if info:
            records.append({
                "symbol": symbol.upper(),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "long_business_summary": (info.get("longBusinessSummary", "") or "")[:2000],
                "company_name": info.get("longName") or info.get("shortName") or symbol.upper(),
                "report_date": pd.Timestamp.today().normalize().date(),
            })
    
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public READ helpers – Direct pandas operations on yfinance data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SHORT)
def load_homepage_data() -> pd.DataFrame:
    """Load company profiles with latest data."""
    symbols = tuple(DEFAULT_TICKERS)
    profile_df = _fetch_batch_tickers(symbols)
    
    if profile_df.empty:
        return pd.DataFrame()
    
    profile_df["symbol"] = profile_df["symbol"].astype(str).str.strip().str.upper()
    profile_df["sector"] = profile_df["sector"].astype(str).str.strip()
    profile_df["industry"] = profile_df["industry"].astype(str).str.strip()
    profile_df = profile_df.dropna(subset=["symbol", "sector"])
    profile_df = (
        profile_df
        .sort_values(["symbol", "report_date"])
        .drop_duplicates(subset=["symbol"], keep="last")
    )
    
    return profile_df.sort_values(["sector", "company_name"]).reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_SHORT)
def load_price_snapshot() -> pd.DataFrame:
    """Get latest price for each symbol."""
    prices_list = []
    for symbol in DEFAULT_TICKERS:
        hist = _fetch_price_history(symbol, period="1y")
        if not hist.empty:
            latest = hist.iloc[-1:].copy()
            prices_list.append(latest)
    
    if not prices_list:
        return pd.DataFrame(columns=["symbol", "report_date", "close"])
    
    prices_df = pd.concat(prices_list, ignore_index=True)
    prices_df["symbol"] = prices_df["symbol"].astype(str).str.strip().str.upper()
    prices_df["report_date"] = pd.to_datetime(prices_df["report_date"], errors="coerce")
    prices_df["close"] = pd.to_numeric(prices_df["close"], errors="coerce")
    prices_df = prices_df.dropna(subset=["symbol", "report_date", "close"])
    return prices_df[["symbol", "report_date", "close"]].reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_SHORT)
def load_latest_price_date() -> str:
    """Return the latest trading date as 'YYYY-MM-DD'."""
    prices = load_price_snapshot()
    if prices.empty or prices["report_date"].isna().all():
        return "—"
    latest = prices["report_date"].max()
    if pd.isna(latest):
        return "—"
    return pd.Timestamp(latest).strftime("%Y-%m-%d")


@st.cache_data(ttl=CACHE_TTL_SHORT, show_spinner=False)
def load_price_history(symbols: tuple[str, ...], lookback_days: int = 1825) -> pd.DataFrame:
    """Daily close prices for given symbols."""
    syms = tuple(s.strip().upper() for s in symbols if s and s.strip())
    if not syms:
        return pd.DataFrame(columns=["symbol", "report_date", "close"])
    
    period_map = {
        1825: "5y",
        730: "2y",
        365: "1y",
        90: "3mo",
        30: "1mo",
    }
    period = period_map.get(lookback_days, "5y")
    
    prices_list = []
    for symbol in syms:
        hist = _fetch_price_history(symbol, period=period)
        if not hist.empty:
            prices_list.append(hist[["symbol", "report_date", "close"]])
    
    if not prices_list:
        return pd.DataFrame(columns=["symbol", "report_date", "close"])
    
    df = pd.concat(prices_list, ignore_index=True)
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["symbol", "report_date", "close"])
    
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
    df = df[df["report_date"] >= cutoff]
    
    return df.sort_values(["symbol", "report_date"]).reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_SHORT)
def load_stock_metrics() -> pd.DataFrame:
    """Load price momentum, volatility, dividends, EPS, and revenue growth."""
    symbols = tuple(DEFAULT_TICKERS)
    metrics_list = []
    
    for symbol in symbols:
        try:
            info = _fetch_ticker_info(symbol)
            hist = _fetch_price_history(symbol, period="5y")
            divs = _fetch_dividends(symbol)
            
            if hist.empty:
                continue
            
            latest_close = hist.iloc[-1]["close"]
            
            # Price momentum (1Y)
            year_ago = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
            year_ago_data = hist[hist["report_date"] <= year_ago]
            price_momentum = None
            if not year_ago_data.empty:
                year_ago_close = year_ago_data.iloc[-1]["close"]
                if year_ago_close != 0:
                    price_momentum = round((latest_close - year_ago_close) / year_ago_close * 100, 2)
            
            # Volatility (90D)
            recent = hist[hist["report_date"] >= pd.Timestamp.utcnow() - pd.Timedelta(days=90)]
            volatility = None
            if not recent.empty and "high" in recent.columns and "low" in recent.columns:
                volatility = round(
                    ((recent["high"] - recent["low"]) / recent["close"].fillna(1)).mean() * 100,
                    2
                )
            
            # Dividends
            pays_dividends = 0 if divs.empty else 1
            
            # EPS
            tailing_eps = info.get("trailingEps")
            
            # Revenue growth
            revenue_growth = None
            try:
                fin = yf.Ticker(symbol).financials
                if fin is not None and not fin.empty:
                    rev_rows = [idx for idx in fin.index if "revenue" in str(idx).lower()]
                    if rev_rows:
                        rev_data = fin.loc[rev_rows[0]].dropna().sort_index(ascending=False)
                        if len(rev_data) >= 2:
                            curr_rev = rev_data.iloc[0]
                            prev_rev = rev_data.iloc[1]
                            if prev_rev != 0:
                                revenue_growth = round((curr_rev - prev_rev) / abs(prev_rev) * 100, 2)
            except Exception:
                pass
            
            metrics_list.append({
                "symbol": symbol.upper(),
                "latest_close": latest_close,
                "price_momentum": price_momentum,
                "volatility": volatility,
                "pays_dividends": pays_dividends,
                "tailing_eps": tailing_eps,
                "revenue_growth": revenue_growth,
            })
        except Exception:
            continue
    
    if not metrics_list:
        return pd.DataFrame()
    
    return pd.DataFrame(metrics_list).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shared helpers used by multiple pages
# ---------------------------------------------------------------------------

def prepare_merged_universe(
    profile_df: pd.DataFrame,
    latest_prices_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    preferred_sector: str,
    wants_dividends: bool,
    prefer_lower_price: bool,
) -> pd.DataFrame:
    """Merge profile, prices, and metrics with filtering."""
    if profile_df.empty or latest_prices_df.empty:
        return pd.DataFrame()

    merged = profile_df.merge(latest_prices_df[["symbol", "close"]], on="symbol", how="inner")
    merged = merged.dropna(subset=["close"])

    if not metrics_df.empty:
        merged = merged.merge(
            metrics_df.drop(columns=["latest_close"], errors="ignore"),
            on="symbol",
            how="left",
        )
    else:
        merged["price_momentum"] = None
        merged["volatility"] = None
        merged["pays_dividends"] = 0
        merged["tailing_eps"] = None
        merged["revenue_growth"] = None

    if preferred_sector and preferred_sector != "Any":
        sector_filtered = merged[merged["sector"] == preferred_sector]
        if not sector_filtered.empty:
            merged = sector_filtered

    if prefer_lower_price:
        median_price = merged["close"].median()
        merged = merged[merged["close"] <= median_price]

    if wants_dividends and "pays_dividends" in merged.columns:
        div_filtered = merged[merged["pays_dividends"] == 1]
        if not div_filtered.empty:
            merged = div_filtered

    return merged.reset_index(drop=True)


def snapshot_for_symbols(
    symbols: list[str],
    profile_df: pd.DataFrame,
    latest_prices_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Get snapshot data for specific symbols."""
    if not symbols or profile_df.empty or latest_prices_df.empty:
        return pd.DataFrame()

    norm = list(dict.fromkeys(str(s).strip().upper() for s in symbols if s.strip()))
    sub = profile_df[profile_df["symbol"].isin(norm)].copy()
    if sub.empty:
        return pd.DataFrame()

    pr = (
        latest_prices_df[latest_prices_df["symbol"].isin(norm)][
            ["symbol", "close", "report_date"]
        ]
        .rename(columns={"report_date": "price_date"})
    )
    out = sub.merge(pr, on="symbol", how="left")

    if not metrics_df.empty:
        out = out.merge(
            metrics_df.drop(columns=["latest_close"], errors="ignore"),
            on="symbol",
            how="left",
        )
    else:
        out["price_momentum"] = None
        out["volatility"] = None
        out["pays_dividends"] = 0
        out["tailing_eps"] = None
        out["revenue_growth"] = None

    order = {s: i for i, s in enumerate(norm)}
    out["_ord"] = out["symbol"].map(order)
    out = out.sort_values("_ord").drop(columns=["_ord"], errors="ignore")
    return out.reset_index(drop=True)


def company_label_options(profile_df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Generate dropdown options for company selection."""
    if profile_df.empty or "symbol" not in profile_df.columns:
        return [], {}

    sub = profile_df[["symbol", "company_name"]].drop_duplicates(subset=["symbol"])
    packed: list[tuple[str, str, str]] = []
    for _, r in sub.iterrows():
        sym = str(r["symbol"]).strip().upper()
        raw_name = str(r.get("company_name", sym) or sym).strip() or sym
        name = raw_name if len(raw_name) <= 90 else raw_name[:87] + "..."
        label = f"{name} ({sym})"
        packed.append((name.lower(), label, sym))

    packed.sort(key=lambda x: x[0])
    options = [x[1] for x in packed]
    label_to_symbol = {x[1]: x[2] for x in packed}
    return options, label_to_symbol


def yahoo_style_comparison_table(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Format snapshot as comparison table."""
    if snapshot.empty:
        return pd.DataFrame()

    cols: dict[str, dict[str, str]] = {}
    for _, row in snapshot.iterrows():
        sym = str(row["symbol"])
        price_date = row.get("price_date")
        date_s = (
            pd.Timestamp(price_date).strftime("%Y-%m-%d")
            if pd.notna(price_date)
            else "—"
        )

        def _fmt_num(val, fmt) -> str:
            try:
                return format(float(val), fmt) if pd.notna(val) else "—"
            except (TypeError, ValueError):
                return "—"

        cols[sym] = {
            "Company": str(row.get("company_name", sym)),
            "Sector": str(row.get("sector", "—")),
            "Industry": str(row.get("industry", "—")),
            "Last price": f"${float(row['close']):,.2f}" if pd.notna(row.get("close")) else "—",
            "Price as of": date_s,
            "1Y price change": f"{float(row['price_momentum']):+.1f}%" if pd.notna(row.get("price_momentum")) else "—",
            "90d volatility (avg daily range)": f"{float(row['volatility']):.1f}%" if pd.notna(row.get("volatility")) else "—",
            "Trailing EPS": _fmt_num(row.get("tailing_eps"), ".2f"),
            "Revenue growth (YoY)": f"{float(row['revenue_growth']):+.1f}%" if pd.notna(row.get("revenue_growth")) else "—",
            "Dividend paid (last 2y)": "Yes" if row.get("pays_dividends", 0) == 1 else "No",
        }

    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Analytics views — Market Intelligence page
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SHORT)
def load_sector_benchmarks() -> pd.DataFrame:
    """Calculate average metrics by sector."""
    profile = load_homepage_data()
    metrics = load_stock_metrics()
    
    if profile.empty or metrics.empty:
        return pd.DataFrame()
    
    merged = profile.merge(metrics, on="symbol", how="left")
    
    sector_stats = merged.groupby("sector").agg({
        "symbol": "count",
        "price_momentum": "mean",
        "volatility": "mean",
    }).reset_index()
    
    sector_stats.columns = ["sector", "symbol_count", "avg_1y_momentum_pct", "avg_90d_volatility_pct"]
    sector_stats = sector_stats.fillna(0).round(2)
    
    return sector_stats.sort_values("symbol_count", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_SHORT)
def load_top_movers(limit: int = 15, losers: bool = False) -> pd.DataFrame:
    """Get top price movers."""
    profile = load_homepage_data()
    metrics = load_stock_metrics()
    
    if metrics.empty:
        return pd.DataFrame()
    
    merged = profile.merge(metrics, on="symbol", how="left")
    merged = merged.dropna(subset=["price_momentum"])
    
    order = "ascending" if losers else "descending"
    merged = merged.sort_values("price_momentum", ascending=losers).head(max(1, int(limit)))
    
    return merged[["company_name", "symbol", "latest_close", "price_momentum"]].reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_MEDIUM)
def load_recent_sec_filings(limit: int = 40) -> pd.DataFrame:
    """Get latest company filings."""
    profile = load_homepage_data()
    lim = max(1, min(int(limit), 500))
    
    result = profile[["symbol", "company_name", "report_date"]].copy()
    result.columns = ["symbol", "company_name", "filing_date"]
    result = result.sort_values("filing_date", ascending=False).head(lim)
    result["filing_date"] = pd.to_datetime(result["filing_date"], errors="coerce")
    
    return result.reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_SHORT)
def load_volume_vs_average(limit: int = 25) -> pd.DataFrame:
    """Get symbols with highest volume spike vs prior 30 sessions."""
    lim = max(1, min(int(limit), 200))
    volume_data = []
    
    for symbol in DEFAULT_TICKERS:
        hist = _fetch_price_history(symbol, period="3mo")
        if hist.empty or len(hist) < 31:
            continue
        
        hist = hist.sort_values("report_date")
        last_volume = hist.iloc[-1]["volume"] if "volume" in hist.columns else 0
        prior_30 = hist.iloc[-31:-1]["volume"].mean() if "volume" in hist.columns else 0
        
        if prior_30 > 0:
            ratio = last_volume / prior_30
            volume_data.append({
                "symbol": symbol,
                "last_day_volume": round(last_volume, 0),
                "avg_30d_volume": round(prior_30, 0),
                "volume_vs_30d_avg": round(ratio, 2),
            })
    
    if not volume_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(volume_data)
    return df.sort_values("volume_vs_30d_avg", ascending=False).head(lim).reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_MEDIUM)
def load_recent_dividend_events(limit: int = 50) -> pd.DataFrame:
    """Get recent dividend payments."""
    lim = max(1, min(int(limit), 300))
    div_list = []
    
    for symbol in DEFAULT_TICKERS:
        divs = _fetch_dividends(symbol)
        if not divs.empty:
            div_list.append(divs)
    
    if not div_list:
        return pd.DataFrame()
    
    df = pd.concat(div_list, ignore_index=True)
    df = df.sort_values("report_date", ascending=False).head(lim)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    
    return df.reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_SHORT)
def load_screener_results(
    *,
    sector: Optional[str] = None,
    min_momentum: Optional[float] = None,
    max_volatility: Optional[float] = None,
    min_eps: Optional[float] = None,
    dividend_only: bool = False,
    limit: int = 200,
) -> pd.DataFrame:
    """Screen stocks based on criteria."""
    lim = max(1, min(int(limit), 500))
    
    profile = load_homepage_data()
    metrics = load_stock_metrics()
    
    if profile.empty or metrics.empty:
        return pd.DataFrame()
    
    base = profile.merge(metrics, on="symbol", how="left")
    
    if sector and str(sector).strip().lower() != "any":
        base = base[base["sector"] == sector]
    
    if min_momentum is not None:
        base = base[base["price_momentum"].fillna(-float("inf")) >= float(min_momentum)]
    
    if max_volatility is not None:
        base = base[base["volatility"].fillna(float("inf")) <= float(max_volatility)]
    
    if min_eps is not None:
        base = base[base["tailing_eps"].fillna(-float("inf")) >= float(min_eps)]
    
    if dividend_only:
        base = base[base["pays_dividends"] == 1]
    
    return base.sort_values("price_momentum", ascending=False, na_position="last").head(lim).reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_LONG)
def load_statement_highlights(limit_symbols: int = 30) -> pd.DataFrame:
    """Load revenue data for top symbols."""
    lim = max(1, min(int(limit_symbols), 200))
    stmt_list = []
    
    for symbol in DEFAULT_TICKERS:
        try:
            ticker = yf.Ticker(symbol)
            fin = ticker.financials
            if fin is not None and not fin.empty:
                rev_rows = [idx for idx in fin.index if "revenue" in str(idx).lower()]
                if rev_rows:
                    rev_data = fin.loc[rev_rows[0]]
                    if not rev_data.empty:
                        latest_date = rev_data.index[0]
                        latest_value = rev_data.iloc[0]
                        stmt_list.append({
                            "symbol": symbol,
                            "item_name": "Total Revenue",
                            "item_value": float(latest_value),
                            "report_date": pd.Timestamp(latest_date),
                            "period_type": "annual",
                        })
        except Exception:
            continue
    
    if not stmt_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(stmt_list)
    df = df.sort_values("item_value", ascending=False, na_position="last").head(lim)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["item_value"] = pd.to_numeric(df["item_value"], errors="coerce")
    
    return df.reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_SHORT)
def load_screener_stocks(limit: int = 2000) -> pd.DataFrame:
    """Load all stocks with screening metrics for Stock Screener page."""
    lim = max(1, min(int(limit), 5000))
    
    profile = load_homepage_data()
    metrics = load_stock_metrics()
    
    if profile.empty or metrics.empty:
        return pd.DataFrame()
    
    base = profile.merge(metrics, on="symbol", how="left")
    
    return base.sort_values("price_momentum", ascending=False, na_position="last").head(lim).reset_index(drop=True)