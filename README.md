# InvestIQ — AI Investment Advisor

A Streamlit application that turns a Databricks Delta lakehouse of US equities
into a **personalised, AI-powered investment workbench**: stock recommendations,
price forecasts, portfolio simulation, goal-based planning, and news
sentiment — all driven by a Groq LLM.


---

## Table of contents

1. [What it does](#what-it-does)
2. [Feature tour](#feature-tour)
3. [Tech stack](#tech-stack)
4. [Architecture](#architecture)
5. [Project layout](#project-layout)
6. [Setup](#setup)
7. [Configuration](#configuration)
8. [Running locally](#running-locally)
9. [Data ingest (one-time)](#data-ingest-one-time)
10. [Deploying to Databricks Apps](#deploying-to-databricks-apps)
11. [Module reference](#module-reference)
12. [Troubleshooting](#troubleshooting)

---

## What it does

A retail investor lands on the home page, sets their risk profile, sector
preference, budget, horizon, and dividend preference, and clicks
**Get AI Recommendations**. The app:

1. Loads the latest fundamentals + prices for ~95 US large-caps from Databricks Delta tables.
2. Filters that universe to candidates that match the profile.
3. Builds a structured prompt and asks **Groq (`llama-3.1-8b-instant` by default)**
   to pick exactly 5 stocks with reasons, a 0–100 confidence score, and three
   risks each.
4. Renders the picks as cards with:
   - sector / industry / current price
   - LLM reason + risk chips + confidence pill
   - on-demand inline **price forecast** (numbers only — current, +30d forecast,
     95% range, model used).
5. Lets the user dive deeper through 8 sub-pages (compare, screen, news,
   portfolio simulator, goal planner, etc.).

The whole experience is themed dark with a fixed cityscape background image
(`assets/hero_bg.png`) and a teal/blue accent palette.

---

## Feature tour

| Page | What you can do | Key tech |
|---|---|---|
| **Home** (`app.py`) | Profile-driven AI stock recommendations, inline price forecast | Groq chat completions, ARIMA + trend fallback |
| **Company Details** (`pages/1`) | Single-ticker deep dive: profile, financials, ARIMA forecast chart, RF buy/hold/avoid signal, RF price regression, news sentiment, AI Q&A | ARIMA, scikit-learn RF (classifier + regressor), VADER sentiment, Groq |
| **All Stocks** (`pages/2`) | Browsable table of the universe with metrics & latest prices | Databricks SQL |
| **Compare Stocks** (`pages/3`) | Side-by-side comparison of 2+ tickers, plus ML signals & sentiment per ticker | RF, VADER |
| **Market Intelligence** (`pages/4`) | Sector breakdown, top movers, volume leaders, recent SEC filings | Databricks SQL views |
| **Stock Screener** (`pages/5`) | Multi-filter SQL screener pushed entirely to the warehouse | Databricks SQL |
| **News Sentiment** (`pages/6`) | Near-real-time Yahoo Finance headlines + VADER scores | yfinance + VADER |
| **Portfolio Simulator** (`pages/7`) | Build a portfolio, see allocation, risk metrics, equity-curve backtest vs equal-weight basket, AI portfolio review | Pandas, Plotly, Groq |
| **Goal Planner** (`pages/8`) | Monte Carlo (5,000 paths) wealth simulation; probability of hitting a goal; required CAGR & SIP top-up; AI action plan | NumPy MC, Groq |

---

## Tech stack

- **Frontend / app server**: [Streamlit](https://streamlit.io/) (multipage)
- **Data warehouse**: Databricks Delta tables under
  `team_tech_innovators.default.*`, queried via the
  [Databricks SQL Statement Execution API](https://docs.databricks.com/api/workspace/statementexecution)
  (no JDBC driver needed).
- **LLM**: [Groq](https://console.groq.com/) — `llama-3.1-8b-instant` for
  chat completions.
- **ML**:
  - [statsmodels](https://www.statsmodels.org/) ARIMA for price forecasting,
    with a log-linear-trend + historical-volatility fallback.
  - [scikit-learn](https://scikit-learn.org/) RandomForest classifier (signal)
    and regressor (price target) in `price_forecast_ml.py`.
- **Sentiment**: [VADER](https://github.com/cjhutto/vaderSentiment) on
  Yahoo Finance headlines pulled through `yfinance`.
- **Charts**: Plotly (Express + Graph Objects).
- **Auth to Databricks**: Databricks SDK **unified authentication** — in many
  orgs, **personal access tokens (PATs) are deprecated**; prefer your team’s
  **service principal** (`sp-team-<name>`) with **OAuth** (`CLIENT_ID` +
  `CLIENT_SECRET`), interactive **CLI OAuth** (`databricks auth login`), or
  env injected in **Databricks Apps**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Streamlit UI                          │
│  app.py + pages/1..8 + theme.py                             │
└───────┬─────────────────────────┬───────────────────────────┘
        │                         │
        │ uses                    │ uses
        ▼                         ▼
┌──────────────────┐   ┌────────────────────────────┐
│ investiq_data.py │   │ llm_chat.py                │
│  - SQL helpers   │   │  - Groq chat completions   │
│  - Caching       │   │                            │
└──────┬───────────┘   └─────────┬──────────────────┘
       │                         │
       ▼                         ▼
┌──────────────────────┐    ┌──────────────────────┐
│ Databricks Delta     │    │ Groq API             │
│  team_tech_inno.     │    │  llama-3.1-8b-instant│
│  default.*           │    │                      │
└──────────────────────┘    └──────────────────────┘
        ▲
        │ one-time / scheduled
        │
┌──────────────────────┐
│ ingest_tickers()     │  ← yfinance → Delta tables
│ (in investiq_data.py)│
└──────────────────────┘
```

Two-phase data design:

- **Ingest phase** (run once or on a schedule, e.g. from a Databricks Job
  or a notebook): `ingest_tickers(symbols)` pulls live data from
  [`yfinance`](https://github.com/ranaroussi/yfinance) and writes it into
  Delta tables. **Streamlit never calls yfinance at request time.**
- **Read phase**: Streamlit pages call `load_*` helpers that run SQL
  against the warehouse. All hot reads are cached with `st.cache_data`.

Tables created/overwritten by ingest:

```
stock_profile          – sector, industry, business summary
stock_prices           – daily OHLCV
stock_sec_filing       – company name + synthetic filing_date
stock_dividend_events  – historical dividends
stock_tailing_eps      – trailing EPS snapshot
stock_statement        – annual income-statement revenue rows
```

---

## Project layout

```
.
├── app.py                       # Streamlit home page
├── theme.py                     # Dark palette + image background injector
├── llm_chat.py                  # Groq chat completion wrapper
├── investiq_data.py             # Databricks SQL data layer + yfinance ingest
├── price_forecast_ml.py         # ARIMA + RF classifier + RF regressor
├── news_sentiment.py            # Databricks / Yahoo news + VADER scoring
├── pages/
│   ├── 1_Company_Details.py
│   ├── 2_All_Stocks.py
│   ├── 3_Compare_Stocks.py
│   ├── 4_Market_Intelligence.py
│   ├── 5_Stock_Screener.py
│   ├── 6_News_Sentiment.py
│   ├── 7_Portfolio_Simulator.py
│   └── 8_Goal_Planner.py
├── assets/
│   └── hero_bg.png              # Background image (base64-embedded at runtime)
├── requirements.txt
├── requirements-ml.txt          # optional: torch + transformers (FinBERT)
├── app.yaml                     # Databricks Apps launch spec
├── .streamlit/
│   └── secrets.toml.example
└── .env.example                 # (if present) sample env vars
```

---

## Setup

### Prerequisites

- Python **3.11 or 3.13** (project tested on 3.13)
- A Databricks workspace with a SQL warehouse and the Delta tables ingested
  (see [Data ingest](#data-ingest-one-time))
- A Groq API key — free at <https://console.groq.com/keys>

### Install

```bash
git clone https://github.com/Team-TechInnovators/AI-InvestmentAdv-Hackathon.git
cd AI-InvestmentAdv-Hackathon

python3 -m venv .venv
source .venv/bin/activate          # on macOS / Linux
# .venv\Scripts\activate           # on Windows

pip install -r requirements.txt
# optional: FinBERT headline sentiment (large download)
# pip install -r requirements-ml.txt
```

> **Important:** if you plan to run `streamlit` from a globally installed
> Python (not the `.venv`), install dependencies into *that* Python instead.
> Mismatched environments are the #1 cause of "ModuleNotFoundError" surprises
> on this project.

---

## Configuration

The app reads two secrets and a few optional knobs.

### 1. `GROQ_API_KEY` (required for AI features)

Pick **one** of these options:

- **`.env` next to `app.py`** (simplest for local dev):

  ```dotenv
  GROQ_API_KEY=gsk_your_key_here
  # Optional: pick a different Groq model
  # GROQ_MODEL=llama-3.1-70b-versatile
  ```

- **`.streamlit/secrets.toml`** (works locally and on Streamlit Community Cloud):

  ```toml
  GROQ_API_KEY = "gsk_your_key_here"
  ```

- **Process env var** (e.g. CI / Databricks Apps):
  set `GROQ_API_KEY` in the deployment environment.

If the key is missing, the home page falls back to a plain filtered list
(no AI reasons / forecasts) and shows a warning.

### 2. Databricks authentication

Many workspaces have **deprecated PATs** for programmatic access in favor of
**service principals + OAuth**. Each team typically gets a principal named
`sp-team-<your-team>`. That principal **does not inherit your user
permissions** — an admin (or you, if you have rights) must grant it access to
the **SQL warehouse** and to **catalog/schema** objects (this app reads
`team_tech_innovators.default.*`).

The Databricks SDK uses **unified authentication** — configure **one** path:

- **Service principal (recommended for local / CI, aligned with org policy):**

  1. In the workspace: **Settings → Identity and access → Service
     principals** — open `sp-team-<your-team>`, copy the **application
     (client) ID**, and **Generate OAuth secret** (secret is shown once).
  2. In `.env`:

  ```dotenv
  DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
  DATABRICKS_CLIENT_ID=<application-uuid>
  DATABRICKS_CLIENT_SECRET=<oauth-secret>
  ```

  Do **not** set `DATABRICKS_TOKEN` when using this flow. Grant the principal
  **USE** on your SQL warehouse and **SELECT** (or equivalent) on the tables
  the app queries.

- **Interactive OAuth via CLI (your user, not the team principal):**

  ```bash
  databricks auth login --host https://<your-workspace>.cloud.databricks.com
  ```

  Then in `.env`:

  ```dotenv
  DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
  DATABRICKS_CONFIG_PROFILE=<profile-name-from-the-login-step>
  ```

  Leave `DATABRICKS_TOKEN`, `DATABRICKS_CLIENT_ID`, and
  `DATABRICKS_CLIENT_SECRET` unset so they do not conflict.

- **Personal access tokens:** only where your org still allows them; PATs are
  **deprecated** in many tenants. If permitted:

  ```dotenv
  DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
  DATABRICKS_TOKEN=dapiXXXXXXXXXXXXXXXXXXXX
  ```

- **Databricks Apps:** the platform injects `DATABRICKS_CLIENT_ID`,
  `DATABRICKS_CLIENT_SECRET`, and `DATABRICKS_HOST` — you normally add only
  app-specific vars (e.g. `GROQ_API_KEY`). Grant the app’s principal warehouse
  and catalog access like any other service principal.

**Genie / LLM help:** if you are unsure how to wire OAuth for an existing
principal, you can ask Databricks Genie (or similar) something like:

> How do I authenticate to Databricks from Python using a service principal’s
> client ID and client secret (OAuth), and what workspace permissions does the
> principal need to run SQL on a warehouse?

### 3. SQL warehouse

The app reads the warehouse id from **`INVESTIQ_SQL_WAREHOUSE_ID`**, or — if
unset — from **`DATABRICKS_SQL_WAREHOUSE_ID`**, matching common Databricks
naming. Only if both are empty does it fall back to a demo id in code; set
one of these explicitly in any shared or production workspace:

```dotenv
INVESTIQ_SQL_WAREHOUSE_ID=<your-warehouse-id>
```

---

## Running locally

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

If you want a quick sanity check that everything is wired up:

```bash
python3 -c "
import importlib
for m in ['app', 'theme', 'llm_chat',
         'investiq_data', 'price_forecast_ml', 'news_sentiment']:
    importlib.import_module(m)
print('All modules import OK')
"
```

---

## Data ingest (one-time)

Before the app can show anything useful, the Delta tables need to be
populated. Open a Databricks notebook (Python) attached to a cluster that
can `pip install yfinance pyarrow`, then:

```python
from investiq_data import ingest_tickers, DEFAULT_TICKERS

# Ingests fundamentals + 5 years of daily prices for the default ~95 tickers.
ingest_tickers(DEFAULT_TICKERS)

# Or your own universe:
# ingest_tickers(["AAPL", "MSFT", "NVDA", "TSLA"])
```

Re-run periodically (e.g. as a daily Databricks Job) to refresh prices and
EPS / revenue / dividend data.

---

## Deploying to Databricks Apps

`app.yaml` already declares the launch command:

```yaml
command: ["streamlit", "run", "app.py"]
env:
  - name: GROQ_API_KEY
    value: "<set-in-databricks-apps-ui>"
```

1. In the Databricks UI, create a new **App** pointing at this repo (or
   sync via the Databricks CLI / Repos).
2. Set secrets and config under the App's *Environment variables* (never
   commit real API keys; keep `app.yaml` placeholders empty or use secret
   scopes):
   - **`GROQ_API_KEY`** — required for LLM features.
   - **`INVESTIQ_SQL_WAREHOUSE_ID`** — SQL warehouse id used for all
     Statement API queries (optional alias: **`DATABRICKS_SQL_WAREHOUSE_ID`**
     if your environment already uses that name).
   - **`INVESTIQ_STOCK_NEWS_FQN`** — optional; fully qualified news view
     name (default `investiq.stock_news`). Must match a view the app principal
     can read.
   - **`GROQ_MODEL`** — optional Groq model override.
3. **Grants for the App service principal** (identity shown on the App settings
   page):
   - **`USE`** on the SQL warehouse above.
   - **`SELECT`** on lakehouse objects the app reads, including at least
     `team_tech_innovators.default.*` (profiles, prices, fundamentals, etc.).
   - **`SELECT`** on the news view (`investiq.stock_news` or your
     `INVESTIQ_STOCK_NEWS_FQN`).
   - Use **Catalog Explorer → Grants** or `GRANT SELECT ON …` in SQL if you
     add new tables or catalogs.
4. **Dependencies:** install from `requirements.txt` for a smaller Apps image
   and faster cold starts. For FinBERT-based headline sentiment (optional),
   add `requirements-ml.txt` (`transformers`, `torch`); the app falls back to
   VADER when those packages are absent.
5. Deploy. The platform injects Databricks **OAuth** for `WorkspaceClient`;
   PATs are not required for the main SQL path or news.

---

## Module reference

### `app.py`
Streamlit home page. Owns the investor profile form, candidate filtering,
LLM prompt construction (`build_recommendation_prompt`) and parsing
(`parse_recommendation_response` — symbol | reason | confidence | risks),
the recommendation card renderer, the inline price-forecast block
(`_render_price_forecast`, with ARIMA → trend fallback), and the navigation
to all sub-pages.

### `theme.py`
Single dark palette declared as CSS custom properties (`--bg`, `--surface`,
`--ink`, `--accent`, …). `inject_theme()` is called from every page *after*
the page's own CSS so the palette wins everywhere. It also lays the
cityscape `assets/hero_bg.png` as a fixed, full-screen background with a
medium dark wash + soft vignette so text stays readable. The image is
base64-embedded into the CSS at runtime (cached with `lru_cache`) — no
static-file serving required.

### `llm_chat.py`
Thin wrapper around the Groq SDK. `chat_completion(messages, model, …)`,
`recommendation_prompt_completion(prompt, model)`, `get_gemini_api_key()`
and `get_gemini_model()` (legacy names; both read `GROQ_*` env vars or
Streamlit secrets). Also auto-loads `.env` on import.

### `investiq_data.py`
Two-phase data layer:

- **Ingest:** `ingest_tickers(symbols)` writes profile, prices, dividends,
  trailing EPS, and revenue rows for each ticker to the Delta tables.
  Runs `yfinance` against each ticker, normalises columns, and uses the
  Databricks SQL Statement Execution API to `MERGE INTO` the tables in
  chunks of 1,000 rows.
- **Read:** `load_homepage_data`, `load_price_snapshot`,
  `load_stock_metrics`, `load_price_history`, `prepare_merged_universe`,
  `load_market_overview`, etc. — every read is cached with `st.cache_data`
  (TTL ranges from 60 s to 5 min).

### `price_forecast_ml.py`
Three models in one file:

1. **ARIMA forecast** — `arima_forecast_confidence(price_df, horizon, confidence)`.
   Auto-selects an order from a small grid; returns point forecast + 95 %
   band. The home page wraps this with a log-linear trend +
   historical-volatility fallback so the demo never shows a hole.
2. **RF classifier** — `run_rf_signal()` produces a Buy / Hold / Avoid
   label based on engineered momentum / volatility / trend features.
3. **RF regressor** — `run_rf_regression()` predicts price 5 / 10 / 20
   trading days ahead.

`build_forecast_chart_figure()` returns a themed Plotly fan-chart used by
the Company Details page.

### `news_sentiment.py`
Reads headlines from a Databricks SQL view via
`investiq_data.run_sql_query_optional` (same **Statement API / OAuth** path
as the rest of the app), then scores with VADER. If the view is missing or
the query fails, falls back to Yahoo Finance via `yfinance.Ticker(symbol).news`
(flattened `content` dict, HTML stripped). Used by `pages/6_News_Sentiment.py`
and on Company Details and Compare pages.

---

## Troubleshooting

**"GROQ_API_KEY not set" warning even though `.env` has it.**
- Make sure you're not running Streamlit from a Python that doesn't see
  your `.env`. Run `which streamlit` and `which python` — they should
  point to the same environment. If they don't,
  `pip install -r requirements.txt` into the Python that owns
  `streamlit`.
- Use Ctrl+C (not Ctrl+Z) to stop Streamlit so you don't end up with
  multiple suspended instances on different ports. `pkill -f
  "streamlit run app.py"` cleans up.

**Databricks authentication errors (credentials, `invalid_client`, PAT, etc.).**
- Prefer **service principal OAuth**: `DATABRICKS_HOST` +
  `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` (non-empty secret).
  Remove or comment **`DATABRICKS_TOKEN`** so it does not override other
  methods.
- If you use **CLI login** instead, set `DATABRICKS_CONFIG_PROFILE` and do
  **not** set `DATABRICKS_CLIENT_ID` / `SECRET` / `TOKEN` unless you mean to.
- **403 / permission** errors after a successful token exchange: grant the
  principal **USE** on the SQL warehouse and **SELECT** on
  `team_tech_innovators.default.*` (principals do not inherit your user ACLs).
- **OAuth `invalid_client` / “Client authentication failed”** — the workspace
  rejected `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET`. Regenerate an
  OAuth **secret** on the **same** service principal as the application ID,
  paste it again into `.env` (no quotes unless required; no extra spaces), restart
  the app. Remove any stray `DATABRICKS_TOKEN` if you intend to use OAuth only.

**"Forecast model could not converge" on a recommendation card.**
- The home-page forecast already has a log-linear trend + volatility-band
  fallback that kicks in automatically. The "Model" pill changes from
  `ARIMA(p,d,q)` to `Trend + vol band` to reflect this. If you see *no*
  forecast at all, the symbol probably has < 30 trading days of history.

**Background image not showing.**
- Check that `assets/hero_bg.png` exists. If you replaced it, hard-refresh
  the browser tab — Streamlit's hot-reload re-imports `theme.py` and the
  `lru_cache` resets.

---

