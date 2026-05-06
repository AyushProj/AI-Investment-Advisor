"""
News headlines + VADER sentiment — primary source from Databricks (Statement API).

Uses the same ``WorkspaceClient`` auth as ``investiq_data`` (OAuth on Databricks Apps).
Falls back to Yahoo Finance if the warehouse view is missing or the query fails.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

import pandas as pd

_STOCK_NEWS_FQN = os.getenv(
    "INVESTIQ_STOCK_NEWS_FQN", "investiq.stock_news"
).strip()
_FQN_SAFE = re.compile(r"^[\w.]+$")

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore[misc, assignment]


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    if BeautifulSoup is not None:
        return BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)
    return re.sub(r"<[^>]+>", " ", raw)


def _score_sentiment(text: str) -> tuple[float, str]:
    """Return (compound, label) via VADER."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    compound = float(analyzer.polarity_scores(text)["compound"])
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return round(compound, 3), label


def _fmt_pub(pub_raw: str) -> str:
    """Normalise a publication timestamp string to 'YYYY-MM-DD HH:MM UTC'."""
    if isinstance(pub_raw, str) and "T" in pub_raw:
        try:
            dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            pass
    return str(pub_raw or "")


def _extract_news_url(val: object) -> str:
    """Resolve Yahoo news link fields (string or ``{\"url\": \"...\"}`` object)."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        u = val.get("url")
        if isinstance(u, str) and u.strip():
            return u.strip()
        for k in ("canonicalUrl", "clickThroughUrl"):
            nested = val.get(k)
            if isinstance(nested, dict):
                u2 = nested.get("url")
                if isinstance(u2, str) and u2.strip():
                    return u2.strip()
    return ""


# ── Primary: Databricks SQL (SDK / OAuth — same path as investiq_data) ────

def _fetch_from_databricks(symbol: str, limit: int) -> pd.DataFrame | None:
    """
    Pull news rows from the configured ``INVESTIQ_STOCK_NEWS_FQN`` view.
    Expected columns: title, summary, published, source, url
    Returns None if auth or SQL fails (caller may fall back to Yahoo).
    """
    if not _STOCK_NEWS_FQN or not _FQN_SAFE.match(_STOCK_NEWS_FQN):
        return None

    from investiq_data import run_sql_query_optional

    sym = "".join(c for c in symbol if c.isalnum() or c in "-._")
    if not sym:
        return None
    sym_esc = sym.replace("'", "''")
    lim = max(1, min(int(limit), 50))
    sql = f"""
        SELECT title, summary, published, source, url
        FROM {_STOCK_NEWS_FQN}
        WHERE UPPER(symbol) = UPPER('{sym_esc}')
        ORDER BY published DESC
        LIMIT {lim}
    """
    df = run_sql_query_optional(sql.strip(), label="stock_news")
    if df is None:
        return None

    records = []
    for _, row in df.iterrows():
        title = str(row.get("title") or "").strip()
        summary = str(row.get("summary") or "").strip()
        if not title:
            continue
        text = f"{title}. {summary}".strip()
        compound, label = _score_sentiment(text)
        records.append(
            {
                "published": _fmt_pub(str(row.get("published") or "")),
                "sentiment": compound,
                "label": label,
                "title": title,
                "source": str(row.get("source") or "—"),
                "url": str(row.get("url") or ""),
            }
        )
    return pd.DataFrame(records)


# ── Fallback: Yahoo Finance ────────────────────────────────────────────────

def _flatten_news_item(item: dict) -> dict:
    """Normalize yfinance news row (nested `content` dict)."""
    c = item.get("content") if isinstance(item.get("content"), dict) else item
    if not isinstance(c, dict):
        return {}
    title = str(c.get("title") or "").strip()
    summary = str(c.get("summary") or "").strip()
    desc = _strip_html(str(c.get("description") or ""))
    if not summary and desc:
        summary = desc[:800]
    pub = c.get("pubDate") or c.get("displayTime") or ""
    prov = c.get("provider")
    if isinstance(prov, dict):
        source = str(prov.get("displayName") or prov.get("sourceId") or "—")
    else:
        source = str(prov or "—")
    url = _extract_news_url(c.get("canonicalUrl")) or _extract_news_url(
        c.get("clickThroughUrl")
    ) or _extract_news_url(c.get("previewUrl"))
    return {
        "title": title,
        "text_for_sentiment": f"{title}. {summary}".strip(),
        "published": pub,
        "source": source,
        "url": url,
    }


def _fetch_from_yfinance(symbol: str, limit: int) -> pd.DataFrame:
    """Fallback: pull headlines from Yahoo Finance via yfinance."""
    import yfinance as yf

    sym = str(symbol or "").strip().upper()
    raw = yf.Ticker(sym).news
    if not raw:
        return pd.DataFrame(
            columns=["published", "sentiment", "label", "title", "source", "url"]
        )

    rows: list[dict] = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        flat = _flatten_news_item(item)
        if not flat.get("title"):
            continue
        text = flat.get("text_for_sentiment") or flat["title"]
        compound, label = _score_sentiment(text)
        rows.append(
            {
                "published": _fmt_pub(str(flat.get("published") or "")),
                "sentiment": compound,
                "label": label,
                "title": flat["title"],
                "source": flat["source"],
                "url": flat["url"],
            }
        )
    return pd.DataFrame(rows)


# ── Public API ────────────────────────────────────────────────────────────

def fetch_yahoo_news_with_sentiment(
    symbol: str, *, limit: int = 25
) -> pd.DataFrame:
    """
    Return a DataFrame of recent news headlines with VADER sentiment scores.

    Data source priority:
      1. Databricks view (``INVESTIQ_STOCK_NEWS_FQN``, default ``investiq.stock_news``)
      2. Yahoo Finance via yfinance (fallback)
    """
    sym = str(symbol or "").strip().upper()
    if not sym:
        return pd.DataFrame()

    lim = max(1, min(int(limit), 50))

    # Try Databricks first
    db_result = _fetch_from_databricks(sym, lim)
    if db_result is not None:
        return db_result

    # Fallback to yfinance
    return _fetch_from_yfinance(sym, lim)