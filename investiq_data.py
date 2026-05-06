"""
InvestIQ data layer.

Two-phase design
----------------
1. INGEST  – pull live data from yfinance and WRITE it into Databricks Delta
             tables under `team_tech_innovators.default.*`.
             Call `ingest_tickers(symbols)` once (e.g. from a notebook or a
             Databricks Job) before running the Streamlit app.

2. READ    – the Streamlit pages call the `load_*` helpers which run SQL
             against those same Delta tables via the Statement Execution API.
             No yfinance calls happen at Streamlit runtime.

Tables created / overwritten by ingest
---------------------------------------
  stock_profile          – sector, industry, business summary
  stock_prices           – daily OHLCV
  stock_sec_filing       – company name + a synthetic "filing_date"
  stock_dividend_events  – historical dividends
  stock_tailing_eps      – trailing EPS snapshot
  stock_statement        – annual income-statement revenue rows
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass


def _normalize_databricks_env_vars() -> None:
    """Normalize Databricks auth env (whitespace, host scheme, CLI vs OAuth)."""
    for key in (
        "DATABRICKS_HOST",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_TOKEN",
        "DATABRICKS_CONFIG_PROFILE",
    ):
        v = os.environ.get(key)
        if v is not None:
            s = v.strip()
            if s != v:
                os.environ[key] = s

    host = os.environ.get("DATABRICKS_HOST", "").strip()
    if host and not host.lower().startswith(("http://", "https://")):
        os.environ["DATABRICKS_HOST"] = f"https://{host.lstrip('/')}"

    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", "").strip()
    cid = os.environ.get("DATABRICKS_CLIENT_ID", "").strip()
    secret = os.environ.get("DATABRICKS_CLIENT_SECRET", "").strip()

    # If CLI profile is set but OAuth secret is missing, drop incomplete OAuth
    # vars so the SDK uses the profile (local dev). Databricks Apps always
    # injects a non-empty secret.
    if profile and cid and not secret:
        del os.environ["DATABRICKS_CLIENT_ID"]
        if "DATABRICKS_CLIENT_SECRET" in os.environ:
            del os.environ["DATABRICKS_CLIENT_SECRET"]


_normalize_databricks_env_vars()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SQL_WAREHOUSE_ID = (
    os.getenv("INVESTIQ_SQL_WAREHOUSE_ID", "").strip()
    or os.getenv("DATABRICKS_SQL_WAREHOUSE_ID", "").strip()
    or "15254d4da5befaaa"
)
CATALOG = "team_tech_innovators"
SCHEMA = "default"
CHUNK_SIZE = 1000

# Default ticker universe (S&P-100 sample).  Override by passing your own
# list to ingest_tickers().
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
# Internal SQL helper
# ---------------------------------------------------------------------------

def get_sql_warehouse_id() -> str:
    return SQL_WAREHOUSE_ID


def is_databricks_auth_failure(exc: BaseException) -> bool:
    """True when ``WorkspaceClient`` cannot authenticate (missing or stale creds)."""
    err = str(exc).lower()
    return any(
        phrase in err
        for phrase in (
            "default credentials",
            "cannot configure",
            "refresh token is invalid",
            "cannot get access token",
            "to reauthenticate",
            "invalid_client",
            "unauthorized_client",
        )
    )


def service_principal_oauth_incomplete() -> bool:
    """True if OAuth M2M was started (client id set) but the client secret is missing.

    PAT or ``DATABRICKS_CONFIG_PROFILE`` counts as an alternate auth path; if set,
    we do not block (see also :func:`_normalize_databricks_env_vars`, which drops
    incomplete OAuth when a profile is present).
    """
    cid = os.getenv("DATABRICKS_CLIENT_ID", "").strip()
    csec = os.getenv("DATABRICKS_CLIENT_SECRET", "").strip()
    if not cid or csec:
        return False
    if os.getenv("DATABRICKS_TOKEN", "").strip():
        return False
    if os.getenv("DATABRICKS_CONFIG_PROFILE", "").strip():
        return False
    return True


def _run_query(w: WorkspaceClient, sql: str, label: str) -> pd.DataFrame:
    """Paginated SQL execution against the configured warehouse."""
    all_rows: list = []
    columns: list[str] | None = None
    offset = 0

    try:
        while True:
            paginated = (
                f"SELECT * FROM ({sql}) _q LIMIT {CHUNK_SIZE} OFFSET {offset}"
            )
            resp = w.statement_execution.execute_statement(
                warehouse_id=SQL_WAREHOUSE_ID,
                statement=paginated,
                wait_timeout="0s",
            )

            while resp.status.state in (
                StatementState.PENDING,
                StatementState.RUNNING,
            ):
                time.sleep(1)
                resp = w.statement_execution.get_statement(resp.statement_id)

            if resp.status.state != StatementState.SUCCEEDED:
                st.error(
                    f"Query '{label}' failed (state={resp.status.state}): "
                    f"{getattr(resp.status, 'error', 'unknown')}"
                )
                break

            if columns is None:
                schema = None
                if (
                    resp.manifest is not None
                    and hasattr(resp.manifest, "schema")
                    and resp.manifest.schema is not None
                ):
                    schema = resp.manifest.schema
                elif (
                    hasattr(resp.result, "schema")
                    and resp.result.schema is not None
                ):
                    schema = resp.result.schema

                if schema is None:
                    st.error(f"Query '{label}': cannot locate schema.")
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
        st.error(f"Exception in query '{label}': {exc}")
        return pd.DataFrame()


def _run_query_optional(
    w: WorkspaceClient, sql: str, label: str
) -> pd.DataFrame | None:
    """Same as _run_query but returns None on auth/SQL errors (no Streamlit UI)."""
    all_rows: list = []
    columns: list[str] | None = None
    offset = 0

    try:
        while True:
            paginated = (
                f"SELECT * FROM ({sql}) _q LIMIT {CHUNK_SIZE} OFFSET {offset}"
            )
            resp = w.statement_execution.execute_statement(
                warehouse_id=SQL_WAREHOUSE_ID,
                statement=paginated,
                wait_timeout="0s",
            )

            while resp.status.state in (
                StatementState.PENDING,
                StatementState.RUNNING,
            ):
                time.sleep(1)
                resp = w.statement_execution.get_statement(resp.statement_id)

            if resp.status.state != StatementState.SUCCEEDED:
                return None

            if columns is None:
                schema = None
                if (
                    resp.manifest is not None
                    and hasattr(resp.manifest, "schema")
                    and resp.manifest.schema is not None
                ):
                    schema = resp.manifest.schema
                elif (
                    hasattr(resp.result, "schema")
                    and resp.result.schema is not None
                ):
                    schema = resp.result.schema

                if schema is None:
                    return None

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

    except Exception:
        return None


def run_sql_query(sql: str, label: str = "query") -> pd.DataFrame:
    """Public helper – run any read-only SQL against the warehouse."""
    return _run_query(WorkspaceClient(), sql, label)


def run_sql_query_optional(sql: str, label: str = "query") -> pd.DataFrame | None:
    """Like ``run_sql_query`` but returns ``None`` on failure (for optional features / fallbacks)."""
    try:
        return _run_query_optional(WorkspaceClient(), sql, label)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# INGEST  — yfinance → Delta tables
# ---------------------------------------------------------------------------

def _tbl(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"


def _exec_ddl(w: WorkspaceClient, sql: str) -> None:
    """Run a DDL/DML statement (no result expected)."""
    resp = w.statement_execution.execute_statement(
        warehouse_id=SQL_WAREHOUSE_ID,
        statement=sql,
        wait_timeout="0s",
    )
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(
            f"DDL failed ({resp.status.state}): "
            f"{getattr(resp.status, 'error', sql[:120])}"
        )


def _insert_df(w: WorkspaceClient, df: pd.DataFrame, table: str) -> None:
    """
    Bulk-insert a DataFrame into a Delta table using multi-row INSERT VALUES.
    Processes in batches of 500 rows to stay within SQL size limits.
    """
    if df.empty:
        return

    cols = list(df.columns)
    col_str = ", ".join(f"`{c}`" for c in cols)
    batch_size = 500

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        rows_sql: list[str] = []
        for _, row in batch.iterrows():
            vals: list[str] = []
            for c in cols:
                v = row[c]
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    vals.append("NULL")
                elif isinstance(v, (int, float)):
                    vals.append(str(v))
                else:
                    escaped = str(v).replace("'", "''").replace("\\", "\\\\")
                    vals.append(f"'{escaped}'")
            rows_sql.append(f"({', '.join(vals)})")

        insert_sql = (
            f"INSERT INTO {_tbl(table)} ({col_str}) VALUES "
            + ", ".join(rows_sql)
        )
        _exec_ddl(w, insert_sql)


def ingest_tickers(
    symbols: list[str] | None = None,
    *,
    price_period: str = "5y",
    truncate: bool = True,
) -> None:
    """
    Fetch data from yfinance for each symbol and write it into Databricks
    Delta tables.  Run this from a Databricks notebook or Job — NOT inside
    the Streamlit app itself.

    Parameters
    ----------
    symbols     : list of ticker strings; defaults to DEFAULT_TICKERS
    price_period: yfinance history period (e.g. "5y", "2y", "1y")
    truncate    : if True, TRUNCATE each table before inserting fresh data
    """
    import yfinance as yf  # only needed at ingest time

    syms = [s.strip().upper() for s in (symbols or DEFAULT_TICKERS) if s.strip()]
    w = WorkspaceClient()

    # ---- ensure tables exist -------------------------------------------------
    ddl_map = {
        "stock_profile": """
            CREATE TABLE IF NOT EXISTS {tbl} (
                symbol          STRING,
                sector          STRING,
                industry        STRING,
                long_business_summary STRING,
                report_date     DATE
            ) USING DELTA
        """,
        "stock_prices": """
            CREATE TABLE IF NOT EXISTS {tbl} (
                symbol      STRING,
                report_date DATE,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                volume      DOUBLE
            ) USING DELTA
        """,
        "stock_sec_filing": """
            CREATE TABLE IF NOT EXISTS {tbl} (
                symbol       STRING,
                company_name STRING,
                filing_date  DATE
            ) USING DELTA
        """,
        "stock_dividend_events": """
            CREATE TABLE IF NOT EXISTS {tbl} (
                symbol      STRING,
                report_date DATE,
                amount      DOUBLE
            ) USING DELTA
        """,
        "stock_tailing_eps": """
            CREATE TABLE IF NOT EXISTS {tbl} (
                symbol      STRING,
                tailing_eps DOUBLE,
                report_date DATE
            ) USING DELTA
        """,
        "stock_statement": """
            CREATE TABLE IF NOT EXISTS {tbl} (
                symbol       STRING,
                item_name    STRING,
                item_value   DOUBLE,
                report_date  DATE,
                period_type  STRING,
                finance_type STRING
            ) USING DELTA
        """,
    }

    for tname, ddl_tmpl in ddl_map.items():
        _exec_ddl(w, ddl_tmpl.format(tbl=_tbl(tname)))

    if truncate:
        for tname in ddl_map:
            try:
                _exec_ddl(w, f"TRUNCATE TABLE {_tbl(tname)}")
            except Exception:
                pass  # table may be empty on first run

    # ---- fetch + insert per ticker -------------------------------------------
    today = pd.Timestamp.today().normalize()

    profile_rows: list[dict] = []
    price_rows: list[dict] = []
    filing_rows: list[dict] = []
    div_rows: list[dict] = []
    eps_rows: list[dict] = []
    stmt_rows: list[dict] = []

    for sym in syms:
        print(f"  Fetching {sym}…")
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}

            # --- profile ------------------------------------------------------
            profile_rows.append(
                {
                    "symbol": sym,
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "long_business_summary": (
                        info.get("longBusinessSummary", "")[:2000]
                    ),
                    "report_date": str(today.date()),
                }
            )

            # --- company name (SEC filing proxy) ------------------------------
            company_name = (
                info.get("longName")
                or info.get("shortName")
                or sym
            )
            filing_rows.append(
                {
                    "symbol": sym,
                    "company_name": company_name,
                    "filing_date": str(today.date()),
                }
            )

            # --- price history ------------------------------------------------
            hist = ticker.history(period=price_period, auto_adjust=True)
            hist.index = pd.to_datetime(hist.index).normalize()
            for dt, row in hist.iterrows():
                price_rows.append(
                    {
                        "symbol": sym,
                        "report_date": str(dt.date()),
                        "open": round(float(row.get("Open", 0) or 0), 6),
                        "high": round(float(row.get("High", 0) or 0), 6),
                        "low": round(float(row.get("Low", 0) or 0), 6),
                        "close": round(float(row.get("Close", 0) or 0), 6),
                        "volume": float(row.get("Volume", 0) or 0),
                    }
                )

            # --- dividends ----------------------------------------------------
            divs = ticker.dividends
            if divs is not None and not divs.empty:
                divs.index = pd.to_datetime(divs.index).normalize()
                for dt, amt in divs.items():
                    div_rows.append(
                        {
                            "symbol": sym,
                            "report_date": str(dt.date()),
                            "amount": round(float(amt), 6),
                        }
                    )

            # --- trailing EPS -------------------------------------------------
            eps_val = info.get("trailingEps")
            if eps_val is not None:
                eps_rows.append(
                    {
                        "symbol": sym,
                        "tailing_eps": round(float(eps_val), 6),
                        "report_date": str(today.date()),
                    }
                )

            # --- annual revenue from financials --------------------------------
            try:
                fin = ticker.financials  # rows = items, cols = dates
                if fin is not None and not fin.empty:
                    rev_row = None
                    for idx in fin.index:
                        if "revenue" in str(idx).lower() or "total revenue" in str(idx).lower():
                            rev_row = fin.loc[idx]
                            break
                    if rev_row is not None:
                        for col in rev_row.index:
                            val = rev_row[col]
                            if pd.notna(val):
                                stmt_rows.append(
                                    {
                                        "symbol": sym,
                                        "item_name": "Total Revenue",
                                        "item_value": float(val),
                                        "report_date": str(
                                            pd.Timestamp(col).date()
                                        ),
                                        "period_type": "annual",
                                        "finance_type": "income_statement",
                                    }
                                )
            except Exception:
                pass  # financials not available for some tickers

        except Exception as exc:
            print(f"  WARNING: could not fetch {sym}: {exc}")
            continue

    # ---- batch insert --------------------------------------------------------
    def _insert(rows: list[dict], table: str) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        _insert_df(w, df, table)

    print("Inserting stock_profile…")
    _insert(profile_rows, "stock_profile")

    print("Inserting stock_prices…")
    _insert(price_rows, "stock_prices")

    print("Inserting stock_sec_filing…")
    _insert(filing_rows, "stock_sec_filing")

    print("Inserting stock_dividend_events…")
    _insert(div_rows, "stock_dividend_events")

    print("Inserting stock_tailing_eps…")
    _insert(eps_rows, "stock_tailing_eps")

    print("Inserting stock_statement…")
    _insert(stmt_rows, "stock_statement")

    print("Ingest complete.")


# ---------------------------------------------------------------------------
# READ helpers  — all data comes from Delta tables via SQL
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_homepage_data() -> pd.DataFrame:
    w = WorkspaceClient()

    query_profile = f"""
        SELECT symbol, sector, industry, long_business_summary, report_date
        FROM {_tbl('stock_profile')}
    """
    query_sec = f"""
        SELECT symbol, company_name, filing_date
        FROM (
            SELECT
                symbol,
                company_name,
                filing_date,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY filing_date DESC) AS rn
            FROM {_tbl('stock_sec_filing')}
        ) ranked
        WHERE rn = 1
    """

    profile_df = _run_query(w, query_profile, "stock_profile")
    if profile_df.empty:
        return pd.DataFrame()

    sec_df = _run_query(w, query_sec, "stock_sec_filing")

    profile_df["symbol"] = profile_df["symbol"].astype(str).str.strip().str.upper()
    profile_df["sector"] = profile_df["sector"].astype(str).str.strip()
    profile_df["industry"] = profile_df["industry"].astype(str).str.strip()
    profile_df["report_date"] = pd.to_datetime(profile_df["report_date"], errors="coerce")
    profile_df = profile_df.dropna(subset=["symbol", "sector"])
    profile_df = (
        profile_df
        .sort_values(["symbol", "report_date"])
        .drop_duplicates(subset=["symbol"], keep="last")
    )

    if sec_df.empty:
        profile_df["company_name"] = profile_df["symbol"]
        return profile_df.sort_values(["sector", "company_name"]).reset_index(drop=True)

    sec_df["symbol"] = sec_df["symbol"].astype(str).str.strip().str.upper()
    sec_df["company_name"] = sec_df["company_name"].astype(str).str.strip()
    sec_df["filing_date"] = pd.to_datetime(sec_df["filing_date"], errors="coerce")
    sec_df = sec_df.dropna(subset=["symbol", "company_name"])
    sec_df = sec_df[sec_df["company_name"] != ""]
    sec_df = sec_df[sec_df["company_name"].str.lower() != "nan"]
    sec_df = (
        sec_df
        .sort_values(["symbol", "filing_date"])
        .drop_duplicates(subset=["symbol"], keep="last")
    )

    df = profile_df.merge(sec_df[["symbol", "company_name"]], on="symbol", how="left")
    df["company_name"] = df["company_name"].fillna(df["symbol"])
    return df.sort_values(["sector", "company_name"]).reset_index(drop=True)


@st.cache_data(ttl=300)
def load_price_snapshot() -> pd.DataFrame:
    w = WorkspaceClient()

    query_prices = f"""
        SELECT symbol, report_date, close
        FROM (
            SELECT
                symbol,
                report_date,
                close,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
            FROM {_tbl('stock_prices')}
        ) ranked
        WHERE rn = 1
    """

    prices_df = _run_query(w, query_prices, "stock_prices")
    if prices_df.empty:
        return pd.DataFrame(columns=["symbol", "report_date", "close"])

    prices_df["symbol"] = prices_df["symbol"].astype(str).str.strip().str.upper()
    prices_df["report_date"] = pd.to_datetime(prices_df["report_date"], errors="coerce")
    prices_df["close"] = pd.to_numeric(prices_df["close"], errors="coerce")
    prices_df = prices_df.dropna(subset=["symbol", "report_date", "close"])
    return prices_df.reset_index(drop=True)


@st.cache_data(ttl=300)
def load_latest_price_date() -> str:
    """
    Return the single latest report_date present in stock_prices as a
    formatted string 'YYYY-MM-DD', or '—' if unavailable.

    This is the authoritative "data freshness" date shown on the home page —
    it reflects the most recent trading day that was successfully ingested
    from yfinance into the Delta table.
    """
    sql = f"SELECT MAX(report_date) AS latest_date FROM {_tbl('stock_prices')}"
    try:
        df = run_sql_query(sql, "latest_price_date")
        if df.empty or df["latest_date"].isna().all():
            return "—"
        val = pd.to_datetime(df["latest_date"].iloc[0], errors="coerce")
        if pd.isna(val):
            return "—"
        return val.strftime("%Y-%m-%d")
    except Exception:
        return "—"


@st.cache_data(ttl=300, show_spinner=False)
def load_price_history(symbols: tuple[str, ...], lookback_days: int = 1825) -> pd.DataFrame:
    """
    Daily close prices for the given symbols over the lookback window.

    Tuple input keeps the result hashable for st.cache_data. Returns a long
    DataFrame with columns: symbol, report_date (datetime64), close (float).
    Empty if no symbols are provided or the warehouse has no rows.
    """
    syms = tuple(s.strip().upper() for s in symbols if s and s.strip())
    if not syms:
        return pd.DataFrame(columns=["symbol", "report_date", "close"])

    in_list = ", ".join(f"'{_sql_quote(s)}'" for s in syms)
    sql = f"""
        SELECT symbol, report_date, close
        FROM {_tbl('stock_prices')}
        WHERE symbol IN ({in_list})
          AND report_date >= DATE_SUB(CURRENT_DATE(), {int(lookback_days)})
    """

    w = WorkspaceClient()
    df = _run_query(w, sql, "stock_prices_history")
    if df.empty:
        return pd.DataFrame(columns=["symbol", "report_date", "close"])

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["symbol", "report_date", "close"])
    return df.sort_values(["symbol", "report_date"]).reset_index(drop=True)


@st.cache_data(ttl=300)
def load_stock_metrics() -> pd.DataFrame:
    w = WorkspaceClient()

    query_price_metrics = f"""
        SELECT
            curr.symbol,
            curr.close AS latest_close,
            prev.close AS close_365d_ago,
            ROUND((curr.close - prev.close) / NULLIF(prev.close, 0) * 100, 2) AS price_momentum,
            vol.volatility
        FROM (
            SELECT symbol, close,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
            FROM {_tbl('stock_prices')}
        ) curr
        LEFT JOIN (
            SELECT symbol, close,
                ROW_NUMBER() OVER (
                    PARTITION BY symbol
                    ORDER BY report_date DESC
                ) AS rn
            FROM {_tbl('stock_prices')}
            WHERE report_date <= DATE_SUB(CURRENT_DATE(), 365)
        ) prev ON curr.symbol = prev.symbol AND prev.rn = 1
        LEFT JOIN (
            SELECT symbol,
                ROUND(AVG((high - low) / NULLIF(close, 0)) * 100, 2) AS volatility
            FROM {_tbl('stock_prices')}
            WHERE report_date >= DATE_SUB(CURRENT_DATE(), 90)
            GROUP BY symbol
        ) vol ON curr.symbol = vol.symbol
        WHERE curr.rn = 1
    """

    query_dividends = f"""
        SELECT DISTINCT symbol, 1 AS pays_dividends
        FROM {_tbl('stock_dividend_events')}
        WHERE report_date >= DATE_SUB(CURRENT_DATE(), 730)
          AND amount > 0
    """

    query_eps = f"""
        SELECT symbol, tailing_eps
        FROM (
            SELECT symbol, tailing_eps,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
            FROM {_tbl('stock_tailing_eps')}
        ) ranked
        WHERE rn = 1
    """

    query_revenue = f"""
        SELECT
            curr.symbol,
            ROUND((curr.item_value - prev.item_value) / NULLIF(ABS(prev.item_value), 0) * 100, 2) AS revenue_growth
        FROM (
            SELECT symbol, item_value, report_date,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
            FROM {_tbl('stock_statement')}
            WHERE LOWER(item_name) LIKE '%total revenue%'
              AND finance_type = 'income_statement'
              AND period_type = 'annual'
        ) curr
        LEFT JOIN (
            SELECT symbol, item_value, report_date,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
            FROM {_tbl('stock_statement')}
            WHERE LOWER(item_name) LIKE '%total revenue%'
              AND finance_type = 'income_statement'
              AND period_type = 'annual'
        ) prev ON curr.symbol = prev.symbol AND prev.rn = 2
        WHERE curr.rn = 1
    """

    price_metrics_df = _run_query(w, query_price_metrics, "price_metrics")
    dividends_df = _run_query(w, query_dividends, "dividends")
    eps_df = _run_query(w, query_eps, "trailing_eps")
    revenue_df = _run_query(w, query_revenue, "revenue_growth")

    if price_metrics_df.empty:
        return pd.DataFrame()

    price_metrics_df["symbol"] = price_metrics_df["symbol"].astype(str).str.strip().str.upper()
    price_metrics_df["price_momentum"] = pd.to_numeric(price_metrics_df["price_momentum"], errors="coerce")
    price_metrics_df["volatility"] = pd.to_numeric(price_metrics_df["volatility"], errors="coerce")
    price_metrics_df["latest_close"] = pd.to_numeric(price_metrics_df["latest_close"], errors="coerce")

    metrics = price_metrics_df[["symbol", "latest_close", "price_momentum", "volatility"]].copy()

    if not dividends_df.empty:
        dividends_df["symbol"] = dividends_df["symbol"].astype(str).str.strip().str.upper()
        dividends_df["pays_dividends"] = (
            pd.to_numeric(dividends_df["pays_dividends"], errors="coerce").fillna(0).astype(int)
        )
        metrics = metrics.merge(dividends_df[["symbol", "pays_dividends"]], on="symbol", how="left")
    else:
        metrics["pays_dividends"] = 0

    metrics["pays_dividends"] = metrics["pays_dividends"].fillna(0).astype(int)

    if not eps_df.empty:
        eps_df["symbol"] = eps_df["symbol"].astype(str).str.strip().str.upper()
        eps_df["tailing_eps"] = pd.to_numeric(eps_df["tailing_eps"], errors="coerce")
        metrics = metrics.merge(eps_df[["symbol", "tailing_eps"]], on="symbol", how="left")
    else:
        metrics["tailing_eps"] = None

    if not revenue_df.empty:
        revenue_df["symbol"] = revenue_df["symbol"].astype(str).str.strip().str.upper()
        revenue_df["revenue_growth"] = pd.to_numeric(revenue_df["revenue_growth"], errors="coerce")
        metrics = metrics.merge(revenue_df[["symbol", "revenue_growth"]], on="symbol", how="left")
    else:
        metrics["revenue_growth"] = None

    return metrics.reset_index(drop=True)


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


def _sql_quote(s: str) -> str:
    return str(s).replace("'", "''")


# ---------------------------------------------------------------------------
# Analytics views — Market Intelligence page
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_sector_benchmarks() -> pd.DataFrame:
    sql = f"""
        WITH prof AS (
            SELECT symbol, sector
            FROM (
                SELECT symbol, sector,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_profile')}
            ) t
            WHERE rn = 1
        ),
        mom AS (
            SELECT
                curr.symbol,
                ROUND((curr.close - prev.close) / NULLIF(prev.close, 0) * 100, 2) AS price_momentum
            FROM (
                SELECT symbol, close,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_prices')}
            ) curr
            LEFT JOIN (
                SELECT symbol, close,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol
                        ORDER BY report_date DESC
                    ) AS rn
                FROM {_tbl('stock_prices')}
                WHERE report_date <= DATE_SUB(CURRENT_DATE(), 365)
            ) prev ON curr.symbol = prev.symbol AND prev.rn = 1
            WHERE curr.rn = 1
        ),
        vol AS (
            SELECT symbol,
                ROUND(AVG((high - low) / NULLIF(close, 0)) * 100, 2) AS volatility
            FROM {_tbl('stock_prices')}
            WHERE report_date >= DATE_SUB(CURRENT_DATE(), 90)
            GROUP BY symbol
        )
        SELECT
            pr.sector,
            COUNT(DISTINCT pr.symbol) AS symbol_count,
            ROUND(AVG(m.price_momentum), 2) AS avg_1y_momentum_pct,
            ROUND(AVG(v.volatility), 2) AS avg_90d_volatility_pct
        FROM prof pr
        LEFT JOIN mom m ON pr.symbol = m.symbol
        LEFT JOIN vol v ON pr.symbol = v.symbol
        WHERE pr.sector IS NOT NULL AND TRIM(CAST(pr.sector AS STRING)) != ''
        GROUP BY pr.sector
        ORDER BY symbol_count DESC
    """
    return run_sql_query(sql, "sector_benchmarks")


@st.cache_data(ttl=300)
def load_top_movers(limit: int = 15, losers: bool = False) -> pd.DataFrame:
    lim = max(1, min(int(limit), 100))
    order = "ASC" if losers else "DESC"
    sql = f"""
        WITH names AS (
            SELECT symbol, company_name
            FROM (
                SELECT symbol, company_name,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY filing_date DESC) AS rn
                FROM {_tbl('stock_sec_filing')}
            ) x
            WHERE rn = 1
        ),
        mom AS (
            SELECT
                curr.symbol,
                ROUND((curr.close - prev.close) / NULLIF(prev.close, 0) * 100, 2) AS price_momentum
            FROM (
                SELECT symbol, close,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_prices')}
            ) curr
            LEFT JOIN (
                SELECT symbol, close,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol
                        ORDER BY report_date DESC
                    ) AS rn
                FROM {_tbl('stock_prices')}
                WHERE report_date <= DATE_SUB(CURRENT_DATE(), 365)
            ) prev ON curr.symbol = prev.symbol AND prev.rn = 1
            WHERE curr.rn = 1
        ),
        priced AS (
            SELECT symbol, close AS last_close
            FROM (
                SELECT symbol, close,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_prices')}
            ) x
            WHERE rn = 1
        )
        SELECT
            COALESCE(n.company_name, m.symbol) AS company_name,
            m.symbol,
            p.last_close,
            m.price_momentum
        FROM mom m
        LEFT JOIN names n ON m.symbol = n.symbol
        LEFT JOIN priced p ON m.symbol = p.symbol
        WHERE m.price_momentum IS NOT NULL
        ORDER BY m.price_momentum {order}
        LIMIT {lim}
    """
    return run_sql_query(sql, "top_movers" if not losers else "bottom_movers")


@st.cache_data(ttl=600)
def load_recent_sec_filings(limit: int = 40) -> pd.DataFrame:
    lim = max(1, min(int(limit), 500))
    sql = f"""
        SELECT symbol, company_name, filing_date
        FROM {_tbl('stock_sec_filing')}
        ORDER BY filing_date DESC
        LIMIT {lim}
    """
    df = run_sql_query(sql, "recent_sec_filings")
    if not df.empty and "filing_date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_volume_vs_average(limit: int = 25) -> pd.DataFrame:
    """Symbols with the highest volume spike vs their prior 30 *sessions*.

    Compares the most recent row in ``stock_prices`` to the average of rows
    ``rn`` 2–31 (31 exclusive), i.e. the previous thirty trading observations.
    Symbols with fewer than two rows are omitted.
    """
    lim = max(1, min(int(limit), 200))
    sql = f"""
        WITH ranked AS (
            SELECT symbol, volume,
                ROW_NUMBER() OVER (
                    PARTITION BY symbol
                    ORDER BY report_date DESC
                ) AS rn
            FROM {_tbl('stock_prices')}
        ),
        last_day AS (
            SELECT symbol, volume AS last_volume
            FROM ranked
            WHERE rn = 1
        ),
        avg_prior AS (
            SELECT symbol, AVG(volume) AS avg_30d_volume
            FROM ranked
            WHERE rn > 1 AND rn <= 31
            GROUP BY symbol
        )
        SELECT
            l.symbol,
            ROUND(l.last_volume, 0) AS last_day_volume,
            ROUND(a.avg_30d_volume, 0) AS avg_30d_volume,
            ROUND(l.last_volume / NULLIF(a.avg_30d_volume, 0), 2) AS volume_vs_30d_avg
        FROM last_day l
        INNER JOIN avg_prior a ON l.symbol = a.symbol
        WHERE a.avg_30d_volume > 0
        ORDER BY volume_vs_30d_avg DESC
        LIMIT {lim}
    """
    return run_sql_query(sql, "volume_vs_avg")


@st.cache_data(ttl=600)
def load_recent_dividend_events(limit: int = 50) -> pd.DataFrame:
    lim = max(1, min(int(limit), 300))
    sql = f"""
        SELECT symbol, report_date, amount
        FROM {_tbl('stock_dividend_events')}
        ORDER BY report_date DESC
        LIMIT {lim}
    """
    df = run_sql_query(sql, "dividend_events")
    if not df.empty and "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    if not df.empty and "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


@st.cache_data(ttl=120)
def load_screener_results(
    *,
    sector: Optional[str] = None,
    min_momentum: Optional[float] = None,
    max_volatility: Optional[float] = None,
    min_eps: Optional[float] = None,
    dividend_only: bool = False,
    limit: int = 200,
) -> pd.DataFrame:
    lim = max(1, min(int(limit), 500))
    where_extra: list[str] = []

    if sector and str(sector).strip() and str(sector).strip().lower() != "any":
        sec = _sql_quote(str(sector).strip())
        where_extra.append(f"TRIM(base.sector) = '{sec}'")

    if min_momentum is not None:
        where_extra.append(f"base.price_momentum >= {float(min_momentum)}")

    if max_volatility is not None:
        where_extra.append(
            f"(base.volatility IS NOT NULL AND base.volatility <= {float(max_volatility)})"
        )

    if min_eps is not None:
        where_extra.append(
            f"(base.tailing_eps IS NOT NULL AND base.tailing_eps >= {float(min_eps)})"
        )

    if dividend_only:
        where_extra.append("base.pays_dividends = 1")

    where_sql = " AND ".join(where_extra) if where_extra else "1=1"

    sql = f"""
        WITH prof AS (
            SELECT symbol, sector, industry
            FROM (
                SELECT symbol, sector, industry,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_profile')}
            ) t
            WHERE rn = 1
        ),
        names AS (
            SELECT symbol, company_name
            FROM (
                SELECT symbol, company_name,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY filing_date DESC) AS rn
                FROM {_tbl('stock_sec_filing')}
            ) x
            WHERE rn = 1
        ),
        price AS (
            SELECT symbol, close AS last_close, report_date AS price_date
            FROM (
                SELECT symbol, close, report_date,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_prices')}
            ) x
            WHERE rn = 1
        ),
        mom AS (
            SELECT
                curr.symbol,
                ROUND((curr.close - prev.close) / NULLIF(prev.close, 0) * 100, 2) AS price_momentum
            FROM (
                SELECT symbol, close,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_prices')}
            ) curr
            LEFT JOIN (
                SELECT symbol, close,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol
                        ORDER BY report_date DESC
                    ) AS rn
                FROM {_tbl('stock_prices')}
                WHERE report_date <= DATE_SUB(CURRENT_DATE(), 365)
            ) prev ON curr.symbol = prev.symbol AND prev.rn = 1
            WHERE curr.rn = 1
        ),
        vol AS (
            SELECT symbol,
                ROUND(AVG((high - low) / NULLIF(close, 0)) * 100, 2) AS volatility
            FROM {_tbl('stock_prices')}
            WHERE report_date >= DATE_SUB(CURRENT_DATE(), 90)
            GROUP BY symbol
        ),
        div AS (
            SELECT DISTINCT symbol, 1 AS pays_dividends
            FROM {_tbl('stock_dividend_events')}
            WHERE report_date >= DATE_SUB(CURRENT_DATE(), 730)
              AND amount > 0
        ),
        eps AS (
            SELECT symbol, tailing_eps
            FROM (
                SELECT symbol, tailing_eps,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_tailing_eps')}
            ) x
            WHERE rn = 1
        ),
        base AS (
            SELECT
                p.symbol, p.sector, p.industry,
                COALESCE(n.company_name, p.symbol) AS company_name,
                pr.last_close, pr.price_date,
                m.price_momentum,
                v.volatility,
                COALESCE(d.pays_dividends, 0) AS pays_dividends,
                e.tailing_eps
            FROM prof p
            INNER JOIN price pr ON p.symbol = pr.symbol
            LEFT JOIN mom   m  ON p.symbol = m.symbol
            LEFT JOIN vol   v  ON p.symbol = v.symbol
            LEFT JOIN div   d  ON p.symbol = d.symbol
            LEFT JOIN eps   e  ON p.symbol = e.symbol
            LEFT JOIN names n  ON p.symbol = n.symbol
        )
        SELECT *
        FROM base
        WHERE {where_sql}
        ORDER BY price_momentum DESC NULLS LAST
        LIMIT {lim}
    """
    return run_sql_query(sql, "screener")


@st.cache_data(ttl=900)
def load_statement_highlights(limit_symbols: int = 30) -> pd.DataFrame:
    lim = max(1, min(int(limit_symbols), 200))
    sql = f"""
        SELECT symbol, item_name, item_value, report_date, period_type
        FROM (
            SELECT symbol, item_name, item_value, report_date, period_type,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
            FROM {_tbl('stock_statement')}
            WHERE LOWER(item_name) LIKE '%total revenue%'
              AND finance_type = 'income_statement'
              AND period_type = 'annual'
        ) x
        WHERE rn = 1
        ORDER BY item_value DESC NULLS LAST
        LIMIT {lim}
    """
    df = run_sql_query(sql, "statement_highlights")
    if not df.empty and "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    if not df.empty and "item_value" in df.columns:
        df["item_value"] = pd.to_numeric(df["item_value"], errors="coerce")
    return df



"""
This function powers the new 5_Stock_Screener.py page.
"""
@st.cache_data(ttl=300)
def load_screener_stocks(limit: int = 2000) -> pd.DataFrame:
    """
    Load all stocks with screening metrics for the Stock Screener page.
 
    Pulls from the same real Delta tables used throughout the rest of the app:
        stock_profile, stock_prices, stock_sec_filing,
        stock_dividend_events, stock_tailing_eps, stock_statement
 
    Returns columns:
        symbol, company_name, sector, industry,
        last_close, price_momentum, volatility,
        pays_dividends, tailing_eps, revenue_growth
    """
    lim = max(1, min(int(limit), 5000))
    sql = f"""
        WITH prof AS (
            SELECT symbol, sector, industry
            FROM (
                SELECT symbol, sector, industry,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_profile')}
            ) t
            WHERE rn = 1
        ),
        names AS (
            SELECT symbol, company_name
            FROM (
                SELECT symbol, company_name,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY filing_date DESC) AS rn
                FROM {_tbl('stock_sec_filing')}
            ) x
            WHERE rn = 1
        ),
        price AS (
            SELECT symbol, close AS last_close
            FROM (
                SELECT symbol, close,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_prices')}
            ) x
            WHERE rn = 1
        ),
        mom AS (
            SELECT
                curr.symbol,
                ROUND((curr.close - prev.close) / NULLIF(prev.close, 0) * 100, 2) AS price_momentum
            FROM (
                SELECT symbol, close,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_prices')}
            ) curr
            LEFT JOIN (
                SELECT symbol, close,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol
                        ORDER BY ABS(DATEDIFF(report_date, DATE_SUB(CURRENT_DATE(), 365)))
                    ) AS rn
                FROM {_tbl('stock_prices')}
            ) prev ON curr.symbol = prev.symbol AND prev.rn = 1
            WHERE curr.rn = 1
        ),
        vol AS (
            SELECT symbol,
                ROUND(AVG((high - low) / NULLIF(close, 0)) * 100, 2) AS volatility
            FROM {_tbl('stock_prices')}
            WHERE report_date >= DATE_SUB(CURRENT_DATE(), 90)
            GROUP BY symbol
        ),
        div AS (
            SELECT DISTINCT symbol, 1 AS pays_dividends
            FROM {_tbl('stock_dividend_events')}
            WHERE report_date >= DATE_SUB(CURRENT_DATE(), 730)
              AND amount > 0
        ),
        eps AS (
            SELECT symbol, tailing_eps
            FROM (
                SELECT symbol, tailing_eps,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_tailing_eps')}
            ) x
            WHERE rn = 1
        ),
        rev AS (
            SELECT
                curr.symbol,
                ROUND(
                    (curr.item_value - prev.item_value) / NULLIF(ABS(prev.item_value), 0) * 100,
                    2
                ) AS revenue_growth
            FROM (
                SELECT symbol, item_value,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_statement')}
                WHERE LOWER(item_name) LIKE '%total revenue%'
                  AND finance_type = 'income_statement'
                  AND period_type  = 'annual'
            ) curr
            LEFT JOIN (
                SELECT symbol, item_value,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
                FROM {_tbl('stock_statement')}
                WHERE LOWER(item_name) LIKE '%total revenue%'
                  AND finance_type = 'income_statement'
                  AND period_type  = 'annual'
            ) prev ON curr.symbol = prev.symbol AND prev.rn = 2
            WHERE curr.rn = 1
        )
        SELECT
            p.symbol,
            COALESCE(n.company_name, p.symbol) AS company_name,
            p.sector,
            p.industry,
            pr.last_close,
            m.price_momentum,
            v.volatility,
            COALESCE(d.pays_dividends, 0)       AS pays_dividends,
            e.tailing_eps,
            r.revenue_growth
        FROM   prof  p
        INNER  JOIN price pr ON p.symbol = pr.symbol
        LEFT   JOIN mom   m  ON p.symbol = m.symbol
        LEFT   JOIN vol   v  ON p.symbol = v.symbol
        LEFT   JOIN div   d  ON p.symbol = d.symbol
        LEFT   JOIN eps   e  ON p.symbol = e.symbol
        LEFT   JOIN rev   r  ON p.symbol = r.symbol
        LEFT   JOIN names n  ON p.symbol = n.symbol
        ORDER  BY m.price_momentum DESC NULLS LAST
        LIMIT  {lim}
    """
    w = WorkspaceClient()                        # same as every other load_* fn
    return _run_query(w, sql, "screener_stocks") # correct 3-arg call