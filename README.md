# Polymarket Edge-Finder (Polymarket Quant Research)

Systematic research framework for detecting statistically testable edges in Polymarket prediction markets.

## What This Includes

- Data pipeline for:
  - Active market metadata (Gamma API)
  - Order books / spreads / liquidity snapshots (CLOB API)
  - Trades and holders (Data API)
  - Resolution outcomes
  - Derived implied probabilities and lagged returns
- Edge research modules:
  - Cross-market inefficiencies
  - Resolution-rule mispricing
  - Liquidity premium
  - Momentum vs mean reversion
  - Whale behavior
- Backtesting engine:
  - Fees + slippage + impact
  - PnL / Sharpe / max drawdown
  - Category-level PnL
- Edge scoring:
  - Significance + capacity + stability
- Streamlit dashboard starter

## Official API Endpoints Used

- Gamma markets: `GET /markets`
- CLOB books: `POST /books`
- CLOB prices: `POST /prices`
- CLOB simplified markets: `GET /sampling-simplified-markets`
- Data API trades: `GET /trades`
- Data API holders: `GET /holders`

## Database Schema

Primary tables created in `polymarket_edge/db.py`:

- `markets`: market metadata + lifecycle fields
- `market_outcomes`: token ids per outcome
- `orderbook_snapshots`: best bid/ask, mid, spread, depth, liquidity score
- `trades`: market trades (wallet-level)
- `holders`: top holders snapshots
- `resolutions`: winner token + resolution timestamp
- `returns`: lagged token returns
- `edge_results`: structured metrics from edge modules
- `backtest_runs`, `backtest_timeseries`, `backtest_trades`: simulation outputs

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
python scripts/bootstrap_db.py
python scripts/run_ingest_once.py --max-markets 800 --trade-markets 100 --resolved-trade-markets 100
python scripts/run_research.py
python -m streamlit run polymarket_edge/dashboard.py
```

For time-series depth (recommended before V2 research), run continuous ingestion:

```bash
python scripts/run_ingest_loop.py --minutes 5 --hours 12 --max-markets 800 --trade-markets 200 --resolved-trade-markets 200
```

Or use the dedicated snapshot collector:

```bash
python scripts/collect_snapshots.py --minutes 5 --hours 24 --max-markets 800
```

For fast timestamp accumulation without trades/resolutions:

```bash
python scripts/run_ingest_loop.py --minutes 0.1 --hours 2 --max-markets 100 --snapshot-only
```

## Research Plan (Practical)

1. Build historical panel
2. Run modules A-E daily after ingestion
3. Rank edges by score (`significance`, `capacity`, `stability`)
4. Backtest only top edges with robust cost model
5. Promote to paper trading
6. Promote to capped live deployment

## Project Structure

```text
polymarket_edge/
  api.py
  config.py
  db.py
  pipeline.py
  features.py
  scoring.py
  dashboard.py
  backtest/
    engine.py
    metrics.py
  edges/
    cross_market.py
    liquidity_premium.py
    momentum_reversion.py
    resolution_rules.py
    whale_behavior.py
scripts/
  bootstrap_db.py
  run_ingest_once.py
  run_research.py
```

## Notes

- This repo focuses on systematic, automatable strategies and excludes illegal/unethical behavior.
- Production deployment should add retries/backoff, monitoring, and stricter execution safeguards.

## Production Research V2

V2 adds independently deployable strategy modules with:

- Unified feature engineering
- Parameterized signals + threshold optimization + model-based alpha
- Independent backtests with multiple holding periods and execution realism
- Statistical validation (walk-forward, bootstrap, t-test, Wilcoxon, stability, overfit gap)
- Advanced validation (permutation test, White reality check, PBO)
- Explainability (feature importance + strongest/weakest conditions)
- Signal diagnostics (calibration, deciles, drift, model-vs-market attribution)
- Capacity estimation
- EV accounting (expected vs realized edge)
- Automated GO/NO-GO deployment gating
- Cross-strategy portfolio analytics (risk parity / mean-variance / max-Sharpe / Kelly)
- Mechanical consistency / parity arbitrage research (YES/NO complement parity + paired execution simulation)

### Actual Edge: Consistency Arbitrage (Mechanical)

This repo now includes a production-style `consistency_arb_v2` strategy focused on structural binary-market constraints:

- Complement parity within a binary market:
  - `ask_yes + ask_no < 1` (buy both legs)
  - `bid_yes + bid_no > 1` (sell both legs, if supported in simulation)
- Paired execution simulation:
  - per-leg fill probabilities
  - pair-fill vs one-leg fill risk
  - unwind penalty for unhedged fills
  - pessimistic execution regime sensitivity

Important:

- The strategy is designed to fail safely when no executable parity violations exist after costs.
- A result of `0` trades is a valid outcome and should be interpreted as “no mechanical arb observed under current filters/cost assumptions”, not as alpha.
- `arb_data_quality.csv` is written before the run and will fail-fast if YES/NO both-leg coverage is too low.

### V2 Run Command

```bash
python scripts/run_research_v2.py
```

Outputs are written to `data/research_v2/`.
`run_research_v2.py` now runs a Research Integrity gate first and fails fast with `insufficient_data_report.csv` when data depth / market coverage / snapshot count is not sufficient.

Run one strategy independently:

```bash
python scripts/run_strategy.py --strategy whale
```

View the V2 dashboard:

```bash
python -m streamlit run polymarket_edge/dashboard_v2.py
```

### V2 Architecture

```text
polymarket_edge/research/
  data.py            # feature panel + targets
  backtest.py        # microstructure-aware multi-horizon simulation
  validation.py      # walk-forward, bootstrap CIs, stability, sensitivity
  diagnostics.py     # calibration, deciles, drift, attribution
  advanced_validation.py  # permutation, White RC, PBO
  deployment.py      # GO/NO-GO readiness checks
  explain.py         # feature importance + condition diagnostics
  capacity.py        # deployable capital estimates
  portfolio.py       # multi-method portfolio optimizer + backtest
polymarket_edge/models/
  base.py            # AlphaModel interface
  logistic.py        # regularized logistic baseline
  gradient_boosting.py
  bayesian.py        # prior+likelihood posterior update model
  evaluation.py      # AUC/logloss/Brier model comparison
  factory.py
polymarket_edge/strategies/
  base.py
  cross_market.py
  resolution_rules.py
  liquidity_premium.py
  momentum.py
  whale.py
scripts/
  run_research_v2.py
  collect_snapshots.py
  backfill_resolved.py
```

### Suggested Parameter Ranges (V2)

- `signal_thresholds`: `[0.05, 0.10, 0.15, 0.20, 0.25]`
- `holding_periods`: `[1, 3, 6, 12]` snapshots
- `model_type`: `heuristic | logistic | gradient_boosting | bayesian | auto | ensemble`
- `model_cv_splits`: `[3, 4, 5]`
- `fee_bps`: `[10, 20, 30]`
- `impact_coeff`: `[0.05, 0.10, 0.20]`
- `max_participation`: `[0.05, 0.10, 0.15, 0.20]`
- `vol_slippage_coeff`: `[0.10, 0.25, 0.50]`
- `fill_beta_spread`: `[1.0, 2.0, 3.0]`
- `train_ratio`: `[0.60, 0.70, 0.80]`

### V2 Performance Report Files

- `strategy_report.csv`: strategy-level outcomes
- `parameter_ranges.csv`: active config by strategy
- `edge_scores_v2.csv`: scored edges (significance/capacity/stability)
- `<strategy>/train_summary.csv`, `<strategy>/test_summary.csv`
- `<strategy>/validation.csv`
- `<strategy>/walkforward_folds.csv`
- `<strategy>/walkforward_summary.csv`
- `<strategy>/stability_time.csv`
- `<strategy>/stability_category.csv`
- `<strategy>/parameter_sensitivity.csv`
- `<strategy>/filter_impact.csv`
- `<strategy>/feature_importance.csv`
- `<strategy>/model_comparison.csv`
- `<strategy>/model_feature_importance.csv`
- `<strategy>/edge_attribution.csv`
- `<strategy>/calibration_curve.csv`
- `<strategy>/signal_deciles.csv`
- `<strategy>/feature_drift.csv`
- `<strategy>/model_vs_market.csv`
- `<strategy>/ev_diagnostics.csv`
- `<strategy>/advanced_validation.csv`
- `<strategy>/deployment_readiness.csv`
- `<strategy>/execution_sensitivity.csv`
- `<strategy>/execution_calibration.csv`
- `<strategy>/strong_conditions.csv`, `<strategy>/weak_conditions.csv`
- `<strategy>/capacity.csv`
- `deployment_readiness.csv` (all strategy criteria rows)
- `deployment_readiness_summary.csv` (one summary row per strategy)
- `portfolio_correlation_<method>.csv`
- `portfolio_weights_<method>.csv`
- `portfolio_summary_<method>.csv`
- `portfolio_summary.csv` (all methods)
- `portfolio_risk_report.csv` (risk parity default risk report)
- `strategy_correlation.csv` (risk parity default strategy correlation)
- `live_opportunities.csv` (top current candidates across strategy + structural modules)
- `consistency_arb_*.csv`, `microstructure_mm_*.csv`, `resolution_mechanics_*.csv`
- `market_links.csv`
- `integrity_report_pre.csv`, `integrity_report_post.csv`
- `insufficient_data_report.csv` (only when integrity gate fails)

### Core APIs

- `run_walkforward(strategy)` in `polymarket_edge/research/validation.py`
- `bootstrap_metrics(returns)` in `polymarket_edge/research/validation.py`
- `build_portfolio(strategies, method=\"risk_parity\")` in `polymarket_edge/research/portfolio.py`

## 7-Day Data Collection (Recommended)

Collect enough time variation before drawing conclusions:

```bash
python scripts/collect_snapshots.py --config configs/collection.yaml
```

Or directly:

```bash
python scripts/collect_snapshots.py --minutes 5 --hours 168 --max-markets 1000 --with-trades --trade-markets 200 --resolved-trade-markets 200
```

## Backfill Resolved Outcomes

```bash
python scripts/backfill_resolved.py --max-rows 5000
```

This populates:

- `resolutions`
- `resolved_markets`
- refreshed `markets` / `market_outcomes`

## Research Integrity + Deploy Gate (How To Interpret)

`PASS` means the strategy satisfied all configured gates on current data:

- positive test mean return
- positive block-bootstrap CI lower bound
- positive walk-forward mean with enough valid folds
- positive EV diagnostics (Spearman + monotonicity)
- survives pessimistic execution assumptions
- no excessive single-market profit concentration

`PASS` does **not** mean guaranteed live profitability. It means the strategy cleared this repo’s current statistical and execution checks and is a candidate for capped paper/live rollout.

If data is insufficient, the run exits early and writes `data/research_v2/insufficient_data_report.csv` instead of producing misleading scores.
