from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from polymarket_edge.backtest.metrics import max_drawdown, sharpe_ratio

INITIAL_CAPITAL = 100_000.0


@dataclass
class PortfolioConfig:
    risk_free_rate: float = 0.0
    max_weight: float = 0.50
    ridge: float = 1e-6
    kelly_cap: float = 0.30
    turnover_penalty: float = 0.01
    total_capital: float = INITIAL_CAPITAL


def _align_returns(returns_by_strategy: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, frame in returns_by_strategy.items():
        if frame.empty or not {"ts", "ret"}.issubset(frame.columns):
            continue
        local = frame[["ts", "ret"]].copy()
        local["ts"] = pd.to_datetime(local["ts"])
        local = local.groupby("ts", observed=True)["ret"].sum().rename(name).reset_index()
        rows.append(local)
    if not rows:
        return pd.DataFrame()
    out = rows[0]
    for nxt in rows[1:]:
        out = out.merge(nxt, on="ts", how="outer")
    out = out.sort_values("ts").fillna(0.0)
    return out


def _cov_matrix(r: pd.DataFrame, ridge: float) -> np.ndarray:
    cov = np.cov(r.T, ddof=0)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = cov + ridge * np.eye(cov.shape[0], dtype=float)
    return cov


def _weights_risk_parity(mu: np.ndarray, cov: np.ndarray, max_weight: float) -> np.ndarray:
    n = len(mu)
    if n == 1:
        return np.array([1.0], dtype=float)
    w0 = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        port_var = float(w @ cov @ w) + 1e-12
        mrc = cov @ w
        rc = w * mrc / np.sqrt(port_var)
        target = np.full(n, rc.sum() / n)
        return float(np.sum((rc - target) ** 2))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, max_weight) for _ in range(n)]
    res = minimize(objective, w0, method="SLSQP", bounds=bnds, constraints=cons)
    if not res.success:
        return w0
    return np.asarray(res.x, dtype=float)


def _weights_mean_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    max_weight: float,
    risk_aversion: float = 6.0,
    *,
    anchor: np.ndarray | None = None,
    turnover_penalty: float = 0.0,
) -> np.ndarray:
    n = len(mu)
    if n == 1:
        return np.array([1.0], dtype=float)
    w0 = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        ret = float(mu @ w)
        var = float(w @ cov @ w)
        penalty = 0.0
        if anchor is not None and turnover_penalty > 0:
            penalty = float(turnover_penalty) * float(np.sum((w - anchor) ** 2))
        return -(ret - 0.5 * risk_aversion * var - penalty)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, max_weight) for _ in range(n)]
    res = minimize(objective, w0, method="SLSQP", bounds=bnds, constraints=cons)
    if not res.success:
        return w0
    return np.asarray(res.x, dtype=float)


def _weights_max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    max_weight: float,
    rf: float = 0.0,
    *,
    anchor: np.ndarray | None = None,
    turnover_penalty: float = 0.0,
) -> np.ndarray:
    n = len(mu)
    if n == 1:
        return np.array([1.0], dtype=float)
    w0 = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        ret = float(mu @ w) - rf
        vol = float(np.sqrt(w @ cov @ w)) + 1e-12
        sharpe = ret / vol
        penalty = 0.0
        if anchor is not None and turnover_penalty > 0:
            penalty = float(turnover_penalty) * float(np.sum((w - anchor) ** 2))
        return -(sharpe - penalty)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, max_weight) for _ in range(n)]
    res = minimize(objective, w0, method="SLSQP", bounds=bnds, constraints=cons)
    if not res.success:
        return w0
    return np.asarray(res.x, dtype=float)


def _weights_kelly(mu: np.ndarray, cov: np.ndarray, max_weight: float, cap: float) -> np.ndarray:
    if len(mu) == 1:
        return np.array([1.0], dtype=float)
    inv = np.linalg.pinv(cov)
    raw = inv @ mu
    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 0:
        w = np.full_like(raw, 1.0 / len(raw))
    else:
        w = raw / raw.sum()
    w = np.minimum(w, cap)
    if w.sum() <= 0:
        return np.full_like(w, 1.0 / len(w))
    w = w / w.sum()
    w = np.minimum(w, max_weight)
    if w.sum() <= 0:
        return np.full_like(w, 1.0 / len(w))
    return w / w.sum()


def _select_weights(method: str, mu: np.ndarray, cov: np.ndarray, cfg: PortfolioConfig) -> np.ndarray:
    key = method.lower().strip()
    anchor = np.full(len(mu), 1.0 / max(1, len(mu)), dtype=float) if len(mu) else np.array([], dtype=float)
    if key in {"risk_parity", "erc", "equal_risk_contribution"}:
        return _weights_risk_parity(mu, cov, cfg.max_weight)
    if key in {"mean_variance", "mv"}:
        return _weights_mean_variance(mu, cov, cfg.max_weight, anchor=anchor, turnover_penalty=cfg.turnover_penalty)
    if key in {"max_sharpe", "sharpe"}:
        return _weights_max_sharpe(mu, cov, cfg.max_weight, rf=cfg.risk_free_rate, anchor=anchor, turnover_penalty=cfg.turnover_penalty)
    if key in {"kelly", "kelly_capped"}:
        return _weights_kelly(mu, cov, cfg.max_weight, cfg.kelly_cap)
    raise ValueError(f"Unknown portfolio method: {method}")


def build_portfolio(
    strategies: dict[str, pd.DataFrame],
    method: str = "risk_parity",
    cfg: PortfolioConfig = PortfolioConfig(),
    *,
    capacity_by_strategy: dict[str, float] | None = None,
    trade_details_by_strategy: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    aligned = _align_returns(strategies)
    if aligned.empty:
        empty = pd.DataFrame()
        return {"correlation": empty, "weights": empty, "timeseries": empty, "summary": empty}

    cols = [c for c in aligned.columns if c != "ts"]
    ret = aligned[cols].to_numpy(dtype=float)
    mu = np.mean(ret, axis=0)
    cov = _cov_matrix(pd.DataFrame(ret, columns=cols), cfg.ridge)
    eff_cfg = PortfolioConfig(
        risk_free_rate=cfg.risk_free_rate,
        max_weight=max(cfg.max_weight, 1.0 / max(1, len(cols))),
        ridge=cfg.ridge,
        kelly_cap=cfg.kelly_cap,
    )
    w = _select_weights(method, mu, cov, eff_cfg)
    w = np.asarray(w, dtype=float)
    capacity_caps = None
    if capacity_by_strategy:
        capacity_caps = np.array(
            [
                float(np.clip((capacity_by_strategy.get(name, np.inf) / max(1.0, float(eff_cfg.total_capital))), 0.0, eff_cfg.max_weight))
                if np.isfinite(capacity_by_strategy.get(name, np.inf))
                else eff_cfg.max_weight
                for name in cols
            ],
            dtype=float,
        )
        # Enforce capacity caps exactly and let unallocated capital sit in cash.
        w = np.minimum(w, capacity_caps)
    gross_alloc = float(np.clip(np.sum(w), 0.0, 1.0))
    if gross_alloc > 1.0:
        w = w / gross_alloc
        gross_alloc = 1.0

    wdf = pd.DataFrame({"strategy": cols, "weight": w})
    if capacity_caps is not None:
        wdf["capacity_weight_cap"] = capacity_caps
        wdf["capacity_bound"] = (w >= (capacity_caps - 1e-10)).astype(int)
    cash_weight = float(max(0.0, 1.0 - float(wdf["weight"].sum()))) if not wdf.empty else 1.0
    wdf = pd.concat([wdf, pd.DataFrame([{"strategy": "cash", "weight": cash_weight}])], ignore_index=True)
    corr = aligned[cols].corr().reset_index().rename(columns={"index": "strategy"})

    aligned["portfolio_ret"] = ret @ w
    aligned["portfolio_equity"] = INITIAL_CAPITAL * (1.0 + aligned["portfolio_ret"]).cumprod()
    aligned["portfolio_drawdown"] = aligned["portfolio_equity"] / aligned["portfolio_equity"].cummax() - 1.0
    n_periods = int(len(aligned))
    portfolio_mean_return = float(aligned["portfolio_ret"].mean()) if n_periods else 0.0
    portfolio_vol = float(aligned["portfolio_ret"].std(ddof=0)) if n_periods else 0.0
    equity_last = float(aligned["portfolio_equity"].iloc[-1]) if n_periods else INITIAL_CAPITAL
    portfolio_total_pnl = float(equity_last - INITIAL_CAPITAL)
    warning = "insufficient_sample_n_periods_lt_30" if n_periods < 30 else ""

    # Invariant: portfolio PnL is always measured versus the fixed initial capital.
    assert np.isclose(
        portfolio_total_pnl,
        equity_last - INITIAL_CAPITAL,
        atol=1e-8,
    ), "Portfolio PnL invariant violated"

    summary = pd.DataFrame(
        [
            {
                "method": method,
                "portfolio_sharpe": sharpe_ratio(aligned["portfolio_ret"]),
                "portfolio_max_drawdown": max_drawdown(aligned["portfolio_equity"]),
                "portfolio_total_pnl": portfolio_total_pnl,
                "portfolio_mean_return": portfolio_mean_return,
                "portfolio_vol": portfolio_vol,
                "n_periods": float(n_periods),
                "warning": warning,
                "n_strategies": float(len(cols)),
                "cash_weight": cash_weight,
                "gross_allocated_weight": float(np.sum(w)),
            }
        ]
    )
    risk_report = _portfolio_risk_report(
        cols=cols,
        weights=w,
        trade_details_by_strategy=trade_details_by_strategy or {},
    )
    return {
        "correlation": corr,
        "strategy_correlation": corr,
        "weights": wdf,
        "timeseries": aligned,
        "summary": summary,
        "risk_report": risk_report,
    }


def build_portfolio_report(returns_by_strategy: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # Backward-compatible wrapper.
    return build_portfolio(returns_by_strategy, method="risk_parity")


def _portfolio_risk_report(
    *,
    cols: list[str],
    weights: np.ndarray,
    trade_details_by_strategy: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    w_ser = pd.Series(weights, index=cols, dtype=float)
    rows.append(
        {
            "risk_type": "strategy_concentration",
            "metric": "hhi_strategy_weights",
            "value": float(np.sum(np.square(w_ser.to_numpy(dtype=float)))) if len(w_ser) else 0.0,
        }
    )
    rows.append(
        {
            "risk_type": "strategy_concentration",
            "metric": "n_active_strategies",
            "value": float((w_ser > 1e-6).sum()),
        }
    )
    if not trade_details_by_strategy:
        return pd.DataFrame(rows)

    market_rows: list[pd.DataFrame] = []
    cat_rows: list[pd.DataFrame] = []
    ttr_rows: list[pd.DataFrame] = []
    for strat, trades in trade_details_by_strategy.items():
        if trades is None or trades.empty or strat not in w_ser.index:
            continue
        w = float(max(0.0, w_ser.get(strat, 0.0)))
        if w <= 0:
            continue
        t = trades.copy()
        notion = pd.to_numeric(t.get("notional"), errors="coerce").fillna(
            pd.to_numeric(t.get("target_notional"), errors="coerce").fillna(0.0)
        ).abs()
        if "market_id" in t.columns:
            tmp_m = pd.DataFrame({"market_id": t["market_id"].astype(str), "exposure": notion * w})
            market_rows.append(tmp_m)
        if "category" in t.columns:
            tmp_c = pd.DataFrame({"category": t["category"].astype(str), "exposure": notion * w})
            cat_rows.append(tmp_c)
        if "time_to_resolution_h" in t.columns:
            ttr = pd.to_numeric(t["time_to_resolution_h"], errors="coerce").fillna(9999.0)
            bucket = pd.Series(
                np.where(ttr <= 24, "<=24h", np.where(ttr <= 72, "24-72h", np.where(ttr <= 168, "3-7d", ">7d"))),
                index=t.index,
            )
            tmp_t = pd.DataFrame({"ttr_bucket": bucket.astype(str), "exposure": notion * w})
            ttr_rows.append(tmp_t)
    if market_rows:
        mkt = pd.concat(market_rows, ignore_index=True).groupby("market_id", observed=True)["exposure"].sum()
        total = float(mkt.sum()) or 1.0
        share = mkt / total
        rows.append({"risk_type": "market_concentration", "metric": "hhi_markets", "value": float(np.sum(np.square(share.to_numpy(dtype=float))))})
        rows.append({"risk_type": "market_concentration", "metric": "top_market_exposure_share", "value": float(share.max()) if len(share) else 0.0})
    if cat_rows:
        cat = pd.concat(cat_rows, ignore_index=True).groupby("category", observed=True)["exposure"].sum()
        total = float(cat.sum()) or 1.0
        for k, v in (cat / total).sort_values(ascending=False).head(10).items():
            rows.append({"risk_type": "category_exposure", "metric": str(k), "value": float(v)})
    if ttr_rows:
        ttr = pd.concat(ttr_rows, ignore_index=True).groupby("ttr_bucket", observed=True)["exposure"].sum()
        total = float(ttr.sum()) or 1.0
        for k, v in (ttr / total).sort_values(ascending=False).items():
            rows.append({"risk_type": "time_to_resolution_exposure", "metric": str(k), "value": float(v)})
    return pd.DataFrame(rows)
