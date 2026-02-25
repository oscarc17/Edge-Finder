from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from polymarket_edge.backtest.metrics import max_drawdown, sharpe_ratio
from polymarket_edge.research.advanced_validation import probability_backtest_overfitting, white_reality_check
from polymarket_edge.research.backtest import StrategyBacktestConfig, run_backtest_grid


@dataclass
class WalkForwardConfig:
    n_folds: int = 4
    min_train_periods: int = 4
    min_train_samples: int = 120
    test_periods: int = 12
    min_test_trades: int = 20
    embargo_periods: int = 1


def bootstrap_metrics(
    returns: pd.Series,
    ts: pd.Series | None = None,
    *,
    n_boot: int = 1000,
    seed: int = 7,
) -> pd.DataFrame:
    vals = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return pd.DataFrame(
            [
                {
                    "mean_return": 0.0,
                    "mean_return_ci_low": 0.0,
                    "mean_return_ci_high": 0.0,
                    "sharpe": 0.0,
                    "sharpe_ci_low": 0.0,
                    "sharpe_ci_high": 0.0,
                    "max_drawdown": 0.0,
                    "max_drawdown_ci_low": 0.0,
                    "max_drawdown_ci_high": 0.0,
                    "mean_return_block": 0.0,
                    "mean_return_block_ci_low": 0.0,
                    "mean_return_block_ci_high": 0.0,
                    "sharpe_block": 0.0,
                    "sharpe_block_ci_low": 0.0,
                    "sharpe_block_ci_high": 0.0,
                }
            ]
        )

    rng = np.random.default_rng(seed)
    mean_draws: list[float] = []
    sharpe_draws: list[float] = []
    dd_draws: list[float] = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        mean_draws.append(float(np.mean(sample)))
        s = pd.Series(sample)
        sharpe_draws.append(sharpe_ratio(s))
        equity = (1.0 + s).cumprod()
        dd_draws.append(max_drawdown(equity))

    def _ci(arr: list[float]) -> tuple[float, float, float]:
        vals_arr = np.array(arr, dtype=float)
        return float(np.mean(vals_arr)), float(np.quantile(vals_arr, 0.025)), float(np.quantile(vals_arr, 0.975))

    mean_mu, mean_lo, mean_hi = _ci(mean_draws)
    shr_mu, shr_lo, shr_hi = _ci(sharpe_draws)
    dd_mu, dd_lo, dd_hi = _ci(dd_draws)

    block_mean_mu = 0.0
    block_mean_lo = 0.0
    block_mean_hi = 0.0
    block_shr_mu = 0.0
    block_shr_lo = 0.0
    block_shr_hi = 0.0
    if ts is not None:
        ts_ser = pd.to_datetime(ts, errors="coerce")
        block_df = pd.DataFrame({"ret": pd.to_numeric(returns, errors="coerce"), "ts": ts_ser}).dropna()
        if not block_df.empty:
            block_df["day"] = block_df["ts"].dt.floor("D")
            by_day = [g["ret"].to_numpy(dtype=float) for _, g in block_df.groupby("day", observed=True)]
            if len(by_day) >= 2:
                rng_block = np.random.default_rng(seed + 13)
                b_mean: list[float] = []
                b_shr: list[float] = []
                for _ in range(n_boot):
                    sample_days = rng_block.integers(0, len(by_day), size=len(by_day))
                    sample = np.concatenate([by_day[int(i)] for i in sample_days]) if len(sample_days) else np.array([], dtype=float)
                    if len(sample) == 0:
                        continue
                    b_mean.append(float(np.mean(sample)))
                    b_shr.append(sharpe_ratio(pd.Series(sample)))
                if b_mean:
                    block_mean_mu, block_mean_lo, block_mean_hi = _ci(b_mean)
                if b_shr:
                    block_shr_mu, block_shr_lo, block_shr_hi = _ci(b_shr)
    return pd.DataFrame(
        [
            {
                "mean_return": mean_mu,
                "mean_return_ci_low": mean_lo,
                "mean_return_ci_high": mean_hi,
                "sharpe": shr_mu,
                "sharpe_ci_low": shr_lo,
                "sharpe_ci_high": shr_hi,
                "max_drawdown": dd_mu,
                "max_drawdown_ci_low": dd_lo,
                "max_drawdown_ci_high": dd_hi,
                "mean_return_block": block_mean_mu,
                "mean_return_block_ci_low": block_mean_lo,
                "mean_return_block_ci_high": block_mean_hi,
                "sharpe_block": block_shr_mu,
                "sharpe_block_ci_low": block_shr_lo,
                "sharpe_block_ci_high": block_shr_hi,
            }
        ]
    )


def _permutation_pvalue(series: pd.Series, *, n_perm: int = 2000, seed: int = 19) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 5:
        return 1.0
    obs = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(vals), replace=True)
        draws.append(float(np.mean(vals * signs)))
    draws_arr = np.asarray(draws, dtype=float)
    return float(np.mean(np.abs(draws_arr) >= abs(obs)))


def _safe_tests(series: pd.Series) -> tuple[float, float, float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 3 or np.std(vals) < 1e-12:
        return 0.0, 1.0, 0.0, 1.0
    t_stat, t_p = stats.ttest_1samp(vals, popmean=0.0, alternative="two-sided")
    try:
        w_stat, w_p = stats.wilcoxon(vals)
    except ValueError:
        w_stat, w_p = 0.0, 1.0
    return float(t_stat), float(t_p), float(w_stat), float(w_p)


def _stability_time(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["ts"] = pd.to_datetime(t["ts"])
    t["period"] = t["ts"].dt.to_period("D").astype(str)
    return (
        t.groupby("period", observed=True)
        .agg(
            n=("trade_return", "count"),
            avg_return=("trade_return", "mean"),
            win_rate=("trade_return", lambda s: float(np.mean(s > 0))),
            sharpe=("trade_return", lambda s: sharpe_ratio(pd.Series(s))),
        )
        .reset_index()
        .sort_values("period")
    )


def _stability_category(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "category" not in trades.columns:
        return pd.DataFrame()
    return (
        trades.groupby("category", observed=True)
        .agg(
            n=("trade_return", "count"),
            avg_return=("trade_return", "mean"),
            win_rate=("trade_return", lambda s: float(np.mean(s > 0))),
            sharpe=("trade_return", lambda s: sharpe_ratio(pd.Series(s))),
        )
        .reset_index()
        .sort_values("avg_return", ascending=False)
    )


def regime_split_performance(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    rows: list[dict[str, float | str]] = []
    for col, group_name in [
        ("prob_vol_6", "volatility"),
        ("liquidity_log", "liquidity"),
        ("time_to_resolution_h", "time_to_resolution"),
    ]:
        if col not in t.columns:
            continue
        x = pd.to_numeric(t[col], errors="coerce")
        r = pd.to_numeric(t.get("trade_return"), errors="coerce")
        valid = x.notna() & r.notna()
        if int(valid.sum()) < 10:
            continue
        x = x.loc[valid]
        r = r.loc[valid]
        if x.nunique() <= 1:
            bucket = pd.Series("all", index=x.index)
        else:
            q1 = float(x.quantile(0.33))
            q2 = float(x.quantile(0.66))
            bucket = pd.Series(np.where(x <= q1, "low", np.where(x <= q2, "mid", "high")), index=x.index)
        tmp = pd.DataFrame({"regime_group": group_name, "regime_bucket": bucket, "trade_return": r})
        grp = (
            tmp.groupby(["regime_group", "regime_bucket"], observed=True)
            .agg(
                n=("trade_return", "count"),
                avg_return=("trade_return", "mean"),
                win_rate=("trade_return", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") > 0))),
                sharpe=("trade_return", lambda s: sharpe_ratio(pd.Series(s))),
            )
            .reset_index()
        )
        rows.extend(grp.to_dict(orient="records"))
    return pd.DataFrame(rows)


def _parameter_heatmap(train_summary_by_fold: pd.DataFrame) -> pd.DataFrame:
    if train_summary_by_fold.empty:
        return pd.DataFrame()
    pivot = (
        train_summary_by_fold.groupby(["threshold", "holding_period"], observed=True)["expectancy"]
        .mean()
        .reset_index()
        .pivot(index="threshold", columns="holding_period", values="expectancy")
        .reset_index()
    )
    return pivot


def _best_row(summary: pd.DataFrame, min_trades: int = 20) -> pd.Series | None:
    if summary.empty:
        return None
    eligible = summary[summary["trade_count"] >= float(min_trades)].copy()
    if eligible.empty:
        eligible = summary[summary["trade_count"] > 0].copy()
    if eligible.empty:
        return None
    score = eligible["expectancy"].fillna(0.0) * np.sqrt(eligible["trade_count"].clip(lower=0.0))
    return eligible.loc[int(score.idxmax())]


def run_walkforward(
    strategy: Any,
    *,
    signal_frame: pd.DataFrame | None = None,
    backtest_config: StrategyBacktestConfig | None = None,
    conn: Any | None = None,
    panel: pd.DataFrame | None = None,
    context: dict[str, Any] | None = None,
    walk_cfg: WalkForwardConfig = WalkForwardConfig(),
) -> dict[str, pd.DataFrame]:
    def _bt_grid(frame: pd.DataFrame, thresholds: list[float], cfg: Any):
        runner = getattr(strategy, "run_backtest_grid", None)
        if callable(runner):
            return runner(frame, thresholds=thresholds, cfg=cfg)
        return run_backtest_grid(frame, thresholds=thresholds, cfg=cfg)

    def _empty_walk_result() -> dict[str, pd.DataFrame]:
        empty_folds = pd.DataFrame(
            columns=[
                "fold",
                "fold_start",
                "fold_end",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "n_trades_train",
                "n_trades_test",
                "label_pos_rate_train",
                "label_y_unique_train",
                "label_mode_train",
                "threshold",
                "holding_period",
                "trade_count",
                "expectancy",
                "sharpe",
                "max_drawdown",
                "valid_fold",
                "status",
                "invalid_reason",
                "embargo_periods",
            ]
        )
        agg = pd.DataFrame(
            [
                {
                    "folds": 0.0,
                    "valid_folds": 0.0,
                    "mean_expectancy": 0.0,
                    "std_expectancy": 0.0,
                    "mean_sharpe": 0.0,
                    "mean_drawdown": 0.0,
                    "mean_trade_count": 0.0,
                    "mean_trades_test": 0.0,
                    "invalid_folds": 0.0,
                    "invalid_reason_counts": "",
                    "embargo_periods": float(getattr(walk_cfg, "embargo_periods", 0)),
                }
            ]
        )
        return {
            "folds": empty_folds,
            "aggregate": agg,
            "stability_time": pd.DataFrame(),
            "stability_category": pd.DataFrame(),
            "parameter_sensitivity": pd.DataFrame(),
            "regime_splits": pd.DataFrame(),
        }

    if signal_frame is None:
        cached = getattr(strategy, "_last_signal_frame", None)
        if isinstance(cached, pd.DataFrame):
            signal_frame = cached
        elif conn is not None and panel is not None:
            signal_frame = strategy.build_signal_frame(conn, panel, context=context)
        else:
            raise ValueError("run_walkforward(strategy) requires cached signal frame or (conn, panel).")
    if backtest_config is None:
        cfg = strategy.config
        backtest_config = StrategyBacktestConfig(
            holding_periods=tuple(cfg.holding_periods),
            fee_bps=cfg.fee_bps,
            impact_coeff=cfg.impact_coeff,
            max_impact=cfg.max_impact,
            base_trade_notional=cfg.base_trade_notional,
            max_participation=cfg.max_participation,
        )

    if signal_frame.empty:
        return _empty_walk_result()

    sf = signal_frame.copy()
    sf["ts"] = pd.to_datetime(sf["ts"], errors="coerce")
    ts_unique = sorted(sf["ts"].dropna().unique())
    embargo = max(0, int(getattr(walk_cfg, "embargo_periods", 0)))
    if len(ts_unique) < walk_cfg.min_train_periods + walk_cfg.test_periods + embargo + 1:
        return _empty_walk_result()

    n_total = len(ts_unique)
    usable_n = n_total - embargo
    train_len = max(walk_cfg.min_train_periods, int(np.floor((usable_n - walk_cfg.test_periods) / max(1, walk_cfg.n_folds))))
    test_len = max(1, walk_cfg.test_periods)
    step = max(1, test_len)

    fold_rows: list[dict[str, float | str | bool | pd.Timestamp]] = []
    all_test_trades: list[pd.DataFrame] = []
    train_summaries: list[pd.DataFrame] = []

    fold_id = 0
    for start in range(0, n_total - train_len - embargo - test_len + 1, step):
        if fold_id >= walk_cfg.n_folds:
            break
        train_ts = ts_unique[start : start + train_len]
        test_start_idx = start + train_len + embargo
        test_ts = ts_unique[test_start_idx : test_start_idx + test_len]
        base_row: dict[str, float | str | bool | pd.Timestamp] = {
            "fold": fold_id,
            "fold_start": pd.Timestamp(train_ts[0]) if len(train_ts) else pd.NaT,
            "fold_end": pd.Timestamp(test_ts[-1]) if len(test_ts) else pd.NaT,
            "train_start": pd.Timestamp(train_ts[0]) if len(train_ts) else pd.NaT,
            "train_end": pd.Timestamp(train_ts[-1]) if len(train_ts) else pd.NaT,
            "test_start": pd.Timestamp(test_ts[0]) if len(test_ts) else pd.NaT,
            "test_end": pd.Timestamp(test_ts[-1]) if len(test_ts) else pd.NaT,
            "n_trades_train": 0.0,
            "n_trades_test": 0.0,
            "label_pos_rate_train": 0.0,
            "label_y_unique_train": 0.0,
            "label_mode_train": "",
            "threshold": 0.0,
            "holding_period": 0.0,
            "trade_count": 0.0,
            "expectancy": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "valid_fold": False,
            "status": "insufficient_data",
            "invalid_reason": "insufficient_data",
            "embargo_periods": float(embargo),
        }
        if len(test_ts) == 0:
            fold_rows.append(base_row)
            fold_id += 1
            continue

        train = sf[sf["ts"].isin(train_ts)].copy()
        test = sf[sf["ts"].isin(test_ts)].copy()
        if train.empty or test.empty:
            fold_rows.append(base_row)
            fold_id += 1
            continue
        if len(train) < int(walk_cfg.min_train_samples):
            base_row["status"] = "insufficient_train_samples"
            base_row["invalid_reason"] = "insufficient_train_samples"
            base_row["n_trades_train"] = float(len(train))
            base_row["n_trades_test"] = float(len(test))
            fold_rows.append(base_row)
            fold_id += 1
            continue

        if hasattr(strategy, "_build_training_labels"):
            try:
                y_fold, diag_fold = strategy._build_training_labels(train)  # type: ignore[attr-defined]
                base_row["label_pos_rate_train"] = float(pd.to_numeric(y_fold, errors="coerce").dropna().mean()) if len(y_fold) else 0.0
                base_row["label_y_unique_train"] = float(pd.to_numeric(y_fold, errors="coerce").dropna().nunique()) if len(y_fold) else 0.0
                if isinstance(diag_fold, pd.DataFrame) and not diag_fold.empty:
                    base_row["label_mode_train"] = str(diag_fold.iloc[-1].get("label_mode", ""))
            except Exception:
                pass

        train_summary, _train_trades = _bt_grid(
            train,
            thresholds=list(strategy.config.signal_thresholds),
            cfg=backtest_config,
        )
        train_summary = train_summary.copy()
        train_summary["fold"] = fold_id
        train_summaries.append(train_summary)
        if train_summary.empty:
            base_row["status"] = "train_backtest_empty"
            base_row["invalid_reason"] = "train_backtest_empty"
            fold_rows.append(base_row)
            fold_id += 1
            continue

        best = _best_row(train_summary, min_trades=strategy.config.min_train_trades)
        if best is None:
            base_row["status"] = "no_train_candidate"
            base_row["invalid_reason"] = "no_train_candidate"
            fold_rows.append(base_row)
            fold_id += 1
            continue
        threshold = float(best["threshold"])
        hp = int(best["holding_period"])
        base_row["threshold"] = threshold
        base_row["holding_period"] = float(hp)
        base_row["n_trades_train"] = float(best.get("trade_count", 0.0))

        test_summary, test_trades = _bt_grid(test, thresholds=[threshold], cfg=backtest_config)
        chosen = test_summary[
            (test_summary["threshold"] == threshold) & (test_summary["holding_period"] == float(hp))
        ]
        if chosen.empty:
            alt = test_summary[test_summary["trade_count"] > 0.0].copy() if not test_summary.empty else pd.DataFrame()
            if not alt.empty:
                alt_score = alt["expectancy"].fillna(0.0) * np.sqrt(alt["trade_count"].clip(lower=0.0))
                chosen = alt.loc[[int(alt_score.idxmax())]]
                hp = int(float(chosen["holding_period"].iloc[0]))
                threshold = float(chosen["threshold"].iloc[0])
                base_row["threshold"] = threshold
                base_row["holding_period"] = float(hp)
        if chosen.empty:
            base_row["status"] = "test_backtest_empty"
            base_row["invalid_reason"] = "test_backtest_empty"
            fold_rows.append(base_row)
            fold_id += 1
            continue

        row = base_row.copy()
        row.update(chosen.iloc[0].to_dict())
        trade_count = float(row.get("trade_count", 0.0))
        row["n_trades_test"] = trade_count
        valid_fold = trade_count >= float(walk_cfg.min_test_trades)
        status = "ok" if valid_fold else "insufficient_test_trades"
        invalid_reason = "" if valid_fold else status
        row.update(
            {
                "threshold": threshold,
                "holding_period": hp,
                "valid_fold": bool(valid_fold),
                "status": status,
                "invalid_reason": invalid_reason,
            }
        )
        fold_rows.append(row)

        if valid_fold:
            fold_trades = test_trades[
                (test_trades["threshold"] == threshold) & (test_trades["holding_period"] == hp)
            ].copy()
            if not fold_trades.empty:
                fold_trades["fold"] = fold_id
                all_test_trades.append(fold_trades)
        fold_id += 1

    folds = pd.DataFrame(fold_rows)
    trades_all = pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame()
    train_grid = pd.concat(train_summaries, ignore_index=True) if train_summaries else pd.DataFrame()

    if folds.empty:
        return _empty_walk_result()

    valid_folds = folds[folds["valid_fold"] == True].copy() if "valid_fold" in folds.columns else folds.copy()  # noqa: E712
    if valid_folds.empty:
        agg = pd.DataFrame(
            [
                {
                    "folds": float(len(folds)),
                    "valid_folds": 0.0,
                    "mean_expectancy": 0.0,
                    "std_expectancy": 0.0,
                    "mean_sharpe": 0.0,
                    "mean_drawdown": 0.0,
                    "mean_trade_count": 0.0,
                    "mean_trades_test": 0.0,
                    "invalid_folds": float(len(folds)),
                    "invalid_reason_counts": ";".join(
                        [
                            f"{k}:{int(v)}"
                            for k, v in folds["status"].astype(str).value_counts(dropna=False).to_dict().items()
                            if str(k) != "ok"
                        ]
                    ),
                    "embargo_periods": float(embargo),
                }
            ]
        )
    else:
        invalid_reasons = ";".join(
            [
                f"{k}:{int(v)}"
                for k, v in folds["status"].astype(str).value_counts(dropna=False).to_dict().items()
                if str(k) != "ok"
            ]
        )
        agg = pd.DataFrame(
            [
                {
                    "folds": float(len(folds)),
                    "valid_folds": float(len(valid_folds)),
                    "mean_expectancy": float(valid_folds["expectancy"].mean()),
                    "std_expectancy": float(valid_folds["expectancy"].std(ddof=0)),
                    "mean_sharpe": float(valid_folds["sharpe"].mean()),
                    "mean_drawdown": float(valid_folds["max_drawdown"].mean()),
                    "mean_trade_count": float(valid_folds["trade_count"].mean()),
                    "mean_trades_test": float(valid_folds["trade_count"].mean()),
                    "invalid_folds": float(len(folds) - len(valid_folds)),
                    "invalid_reason_counts": invalid_reasons,
                    "embargo_periods": float(embargo),
                }
            ]
        )
    return {
        "folds": folds,
        "aggregate": agg,
        "stability_time": _stability_time(trades_all),
        "stability_category": _stability_category(trades_all),
        "parameter_sensitivity": _parameter_heatmap(train_grid),
        "regime_splits": regime_split_performance(trades_all),
    }


def validate_strategy(train_trades: pd.DataFrame, test_trades: pd.DataFrame) -> pd.DataFrame:
    train_ret = train_trades["trade_return"] if not train_trades.empty else pd.Series(dtype=float)
    test_ret = test_trades["trade_return"] if not test_trades.empty else pd.Series(dtype=float)
    test_ts = test_trades["ts"] if (not test_trades.empty and "ts" in test_trades.columns) else None
    boot = bootstrap_metrics(test_ret, ts=test_ts)
    t_stat, t_p, w_stat, w_p = _safe_tests(test_ret)
    perm_p = _permutation_pvalue(test_ret, n_perm=2000, seed=19)
    test_vals = pd.to_numeric(test_ret, errors="coerce").dropna().to_numpy(dtype=float)
    if len(test_vals) >= 2 and np.isfinite(t_stat):
        t_p_pos = float(stats.t.sf(float(t_stat), df=len(test_vals) - 1))
    else:
        t_p_pos = 1.0

    train_mean = float(pd.to_numeric(train_ret, errors="coerce").dropna().mean()) if len(train_ret) else 0.0
    test_mean = float(pd.to_numeric(test_ret, errors="coerce").dropna().mean()) if len(test_ret) else 0.0
    overfit_gap = train_mean - test_mean

    row = boot.iloc[0].to_dict()
    if "mean_return_block_ci_low" in row:
        row["mean_return_ci_low_iid"] = float(row.get("mean_return_ci_low", 0.0))
        row["mean_return_ci_high_iid"] = float(row.get("mean_return_ci_high", 0.0))
        row["mean_return_ci_low"] = float(min(row.get("mean_return_ci_low", 0.0), row.get("mean_return_block_ci_low", 0.0)))
        row["mean_return_ci_high"] = float(max(row.get("mean_return_ci_high", 0.0), row.get("mean_return_block_ci_high", 0.0)))
    row.update(
        {
            "train_mean_return": train_mean,
            "test_mean_return": test_mean,
            "ttest_stat": t_stat,
            "ttest_pvalue": t_p,
            "ttest_pvalue_pos": t_p_pos,
            "wilcoxon_stat": w_stat,
            "wilcoxon_pvalue": w_p,
            "permutation_pvalue": perm_p,
            "n_obs": float(len(test_vals)),
            "overfit_gap": overfit_gap,
            "stability_score": float(max(0.0, 1.0 - abs(overfit_gap))),
        }
    )
    return pd.DataFrame([row])


def block_bootstrap_metrics(
    returns: pd.Series,
    ts: pd.Series,
    *,
    n_boot: int = 1000,
    seed: int = 7,
) -> pd.DataFrame:
    return bootstrap_metrics(returns, ts=ts, n_boot=n_boot, seed=seed)


def run_white_reality_check(candidate_returns: pd.DataFrame, *, n_boot: int = 1000, seed: int = 17) -> pd.DataFrame:
    return pd.DataFrame([white_reality_check(candidate_returns, n_boot=n_boot, seed=seed)])


def run_pbo(candidate_returns: pd.DataFrame, *, n_slices: int = 8, max_trials: int = 200, seed: int = 23) -> pd.DataFrame:
    return pd.DataFrame(
        [
            probability_backtest_overfitting(
                candidate_returns,
                n_slices=n_slices,
                max_trials=max_trials,
                seed=seed,
            )
        ]
    )
