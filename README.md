# Polymarket Edge Finder (Polymarket Quant Research Scaffold)

Edge-Finder is a Python research framework for systematic Polymarket strategy research with a DuckDB warehouse, execution-aware backtests, validation gates, and dashboards.

This repo is designed to reject weak strategies. A `FAIL`, `NO EDGE`, `INCONCLUSIVE`, or `0 trades` result is a valid research outcome.

## What This Repo Does

- Ingests Polymarket data from:
  - Gamma markets metadata
  - CLOB order books / prices
  - Data API trades and holders
  - Simplified markets (resolution truth / winner flags)
- Stores normalized data in DuckDB (`markets`, `market_outcomes`, `orderbook_snapshots`, `trades`, `holders`, `resolutions`, etc.)
- Builds time-series feature panels with time-aware targets
- Runs multiple research strategies (behavioral + structural)
- Simulates execution costs (spread, impact, slippage, fees) and fill models
- Runs validation (walk-forward, bootstrap, diagnostics, deployment gates)
- Produces V2 artifacts and Streamlit dashboards

## What It Does Not Do

- It does not guarantee live profitability.
- It does not assume mid-price fills.
- It does not weaken validation gates to force a positive result.

## Core Research Principles (Implemented)

- Time-aware returns and targets (not row-lag assumptions)
- Explicit resolution truth preferred over heuristics
- Fee-aware expected value and backtests (observed fee rates when available, otherwise fee regime curves/fallback)
- Structural edge research (complement parity / consistency arb) with paired-leg backtests
- Strict GO/NO-GO gating

## Data Sources and API Endpoints

- Gamma API:
  - `GET /markets`
- CLOB API:
  - `POST /books`
  - `POST /prices`
  - `GET /prices-history`
  - `GET /simplified-markets` (primary resolution truth input)
  - `GET /sampling-simplified-markets` (optional sampling endpoint)
- Data API:
  - `GET /trades`
  - `GET /holders`

## Database Schema (High Level)

Defined in `polymarket_edge/db.py`.

Key tables:

- `markets`
  - market metadata and lifecycle fields
  - fee metadata fields such as `fee_enabled`, `fee_regime`, `fee_regime_source`
- `market_outcomes`
  - token ids / labels / winner flags by outcome
- `orderbook_snapshots`
  - best bid/ask, mid, spread, depth, liquidity score by token and timestamp
- `trades`
  - market trades with normalized fields
  - includes `fee_rate_bps` and `fee_rate_source` when present in API payloads
- `holders`
  - periodic holder snapshots
- `resolutions`
  - winner token, resolved timestamp, source, and `label_source` (`explicit` or `inferred`)
- `returns`
  - time-aware lagged returns (`1h`, `6h`, `24h`)

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/bootstrap_db.py
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/bootstrap_db.py
```

## Quick Start (V1 + V2)

1. Ingest a one-off snapshot + some trades/holders
2. Refresh resolution truth
3. Run V2 research
4. Open dashboard

```powershell
python scripts/run_ingest_once.py --max-markets 800 --trade-markets 100 --resolved-trade-markets 100
python scripts/refresh_resolutions.py --max-rows 5000
python scripts/backfill_resolved.py --max-rows 5000
python scripts/run_research_v2.py
python -m streamlit run polymarket_edge/dashboard_v2.py
```

If you want the older pipeline:

```powershell
python scripts/run_research.py
python -m streamlit run polymarket_edge/dashboard.py
```

## Recommended Research Workflow (V2)

This is the workflow other users should follow.

### 1) Collect enough time-series data

The V2 pipeline has an integrity gate. If you do not have enough timestamps / markets / snapshots (or enough YES/NO two-leg coverage for parity research), the run will fail fast and write `insufficient_data_report.csv`.

Snapshot collection (recommended):

```powershell
python scripts/collect_snapshots.py --config configs/collection.yaml
```

Direct collector command (example):

```powershell
python scripts/collect_snapshots.py --minutes 5 --hours 24 --max-markets 1000 --with-trades --trade-markets 200 --resolved-trade-markets 200
```

Fast dev iteration (likely insufficient for full V2, but useful for smoke tests):

```powershell
python scripts/collect_snapshots.py --minutes 0.5 --hours 1 --max-markets 300 --max-cycles 120
```

### 2) Refresh resolution truth (explicit winner flags)

This uses `GET /simplified-markets` and stores explicit winner truth when available.

```powershell
python scripts/refresh_resolutions.py --max-rows 5000
```

Artifacts written:

- `data/research_v2/resolution_truth_audit.csv`
- `data/research_v2/resolution_winner_flag_exceptions.csv`

### 3) Backfill resolved markets

```powershell
python scripts/backfill_resolved.py --max-rows 5000
```

This refreshes:

- `resolutions`
- `resolved_markets`
- `markets`
- `market_outcomes`

### 4) Run V2 research

```powershell
python scripts/run_research_v2.py
```

Outputs are written to `data/research_v2/`.

### 5) Inspect results in the dashboard

```powershell
python -m streamlit run polymarket_edge/dashboard_v2.py
```

## Standalone Strategy Runs

You can run one strategy independently and write artifacts to `data/research_v2_single/<strategy>/`.

```powershell
python scripts/run_strategy.py --strategy consistency_arb --model-type heuristic
python scripts/run_strategy.py --strategy consistency_arb_maker --model-type heuristic
python scripts/run_strategy.py --strategy whale --model-type auto
```

Supported `--strategy` values:

- `consistency_arb`
- `consistency_arb_maker`
- `cross_market`
- `resolution_rules`
- `liquidity_premium`
- `momentum`
- `whale`

## Structural Edge Modules (Current Focus)

### Complement Parity / Consistency Arb

This repo includes structural parity research and execution-aware backtests:

- `polymarket_edge/edges/consistency_arb.py`
  - generates complement parity and related constraint candidates
- `polymarket_edge/strategies/consistency_arb.py` (`consistency_arb_v2`)
  - paired taker-style parity backtest with leg-risk modeling
- `polymarket_edge/strategies/consistency_arb_maker.py` (`consistency_arb_maker_v1`)
  - maker/passive parity research with conservative fill simulation and hedge/bailout logic

Important:

- `0 trades` is a valid outcome if no executable opportunities survive costs / fill assumptions.
- This should be interpreted as "no observed structural arb under current filters and execution model", not as a pipeline bug.

## V2 Artifacts and Reports

### Top-Level (`data/research_v2/`)

- `strategy_report.csv`
- `edge_scores_v2.csv`
- `deployment_readiness.csv`
- `deployment_readiness_summary.csv`
- `readiness_report.csv`
- `live_opportunities.csv`
- `integrity_report_pre.csv`, `integrity_report_post.csv`
- `insufficient_data_report.csv` (only when the integrity gate fails)
- `fee_regime_coverage.csv`
- `sampling_interval_audit.csv`
- `resolution_truth_audit.csv`
- `market_links.csv`, `market_links_summary.csv`
- `consistency_arb_*.csv`, `microstructure_mm_*.csv`, `resolution_mechanics_*.csv`

### Per-Strategy Folder (`data/research_v2/<strategy>/`)

Examples:

- `signals.csv`
- `train_trades.csv`, `test_trades.csv`
- `train_summary.csv`, `test_summary.csv`
- `validation.csv`
- `walkforward_folds.csv`, `walkforward_summary.csv`
- `advanced_validation.csv`
- `ev_diagnostics.csv`
- `execution_sensitivity.csv`
- `execution_calibration.csv`
- `cost_decomposition.csv`
- `deployment_readiness.csv`
- `capacity.csv`
- `feature_importance.csv`

Maker parity strategies can also produce:

- `candidates.csv`
- `paired_trades.csv`
- `completed_sets.csv`
- `maker_fill_diagnostics.csv`
- `maker_fill_diagnostics_summary.csv`

## Research Integrity and Deployment Gates (How to Read Results)

### PASS means

The strategy passed the configured statistical and execution checks on current data, including checks like:

- positive out-of-sample mean return
- positive CI lower bound (bootstrap / block bootstrap where applicable)
- positive walk-forward mean with enough valid folds
- EV diagnostics alignment (e.g., Spearman / monotonicity)
- survival under pessimistic execution assumptions
- no extreme profit concentration in one market

### PASS does not mean

- guaranteed live profitability
- infinite capacity
- no regime risk

It means the strategy is a candidate for controlled paper trading / capped live experimentation.

### FAIL / NO EDGE / INCONCLUSIVE

These are valid outputs.

- `FAIL` / `NO EDGE`
  - The edge did not survive costs, execution realism, or OOS validation.
- `INCONCLUSIVE`
  - Usually means not enough trades, no executed fills, or insufficient data to estimate robustness.

## Timebase and Label Semantics (Important)

- Returns and forward targets are time-based, not row-based.
- `holding_period` and target columns are interpreted using actual timestamps (irregular sampling supported).
- Resolution labels default to `label_source='explicit'` when training on outcomes.
- Inferred labels exist for fallback / audit but are not the default source of truth.

## Fees and Cost Modeling (Important)

- Do not assume a single global fee.
- V2 prefers observed `fee_rate_bps` from trade payloads when available.
- Otherwise it uses fee regime curves (`FEE_FREE`, `CRYPTO_5_15_MIN`, `SPORTS_FEE_CURVE`, etc.).
- When fees are unknown, the code falls back conservatively and logs warnings in relevant paths.

## Troubleshooting

### `streamlit` is not recognized

Use:

```powershell
python -m streamlit run polymarket_edge/dashboard_v2.py
```

### DuckDB file lock (`Cannot open file ... polymarket.duckdb`)

Another process is using the DB (collector, notebook, dashboard, or another research run).

Close the process and rerun:

```powershell
python scripts/run_research_v2.py
```

### V2 exits early with `insufficient_data_report.csv`

You do not have enough data for a meaningful run.

Collect more snapshots and rerun:

```powershell
python scripts/collect_snapshots.py --minutes 5 --hours 24 --max-markets 1000 --with-trades --trade-markets 200 --resolved-trade-markets 200
python scripts/run_research_v2.py
```

### Collector crashes on network/API disconnects

Transient API/network failures happen during long runs. Re-run the collector; existing snapshots already written to DuckDB are preserved.

## Repro / Validation Commands

Run tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Run a quick syntax check:

```powershell
python -m py_compile polymarket_edge/db.py polymarket_edge/pipeline.py polymarket_edge/research/backtest.py
```

## Project Structure (High-Level)

```text
polymarket_edge/
  api.py
  config.py
  db.py
  pipeline.py
  fees.py
  scoring.py
  dashboard.py
  dashboard_v2.py
  backtest/
    engine.py
    metrics.py
  edges/
    consistency_arb.py
    cross_market.py
    liquidity_premium.py
    microstructure_mm.py
    momentum_reversion.py
    resolution_rules.py
    resolution_mechanics.py
    whale_behavior.py
  research/
    data.py
    backtest.py
    paired_backtest.py
    validation.py
    diagnostics.py
    deployment.py
    integrity.py
    maker_execution.py
    maker_parity_backtest.py
  strategies/
    base.py
    consistency_arb.py
    consistency_arb_maker.py
    cross_market.py
    liquidity_premium.py
    momentum.py
    resolution_rules.py
    whale.py
scripts/
  bootstrap_db.py
  collect_snapshots.py
  refresh_resolutions.py
  backfill_resolved.py
  run_ingest_once.py
  run_ingest_loop.py
  run_research.py
  run_research_v2.py
  run_strategy.py
```

## Safety / Ethics

- This repo is for systematic, automatable research.
- Do not use manipulation, insider information, ToS circumvention, or rule gaming.
- Live deployment should add monitoring, retries/backoff, kill switches, exposure limits, and audit logging.
