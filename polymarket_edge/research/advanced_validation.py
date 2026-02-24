from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def _candidate_key(threshold: float, holding_period: int | float) -> str:
    return f"thr={float(threshold):.4f}|hp={int(holding_period)}"


def _candidate_return_matrix(trades: pd.DataFrame) -> pd.DataFrame:
    needed = {"ts", "threshold", "holding_period", "trade_return"}
    if trades.empty or not needed.issubset(trades.columns):
        return pd.DataFrame()
    t = trades.copy()
    t["ts"] = pd.to_datetime(t["ts"])
    t["candidate"] = [
        _candidate_key(thr, hp)
        for thr, hp in zip(
            pd.to_numeric(t["threshold"], errors="coerce").fillna(0.0),
            pd.to_numeric(t["holding_period"], errors="coerce").fillna(0.0),
        )
    ]
    by_ts = (
        t.groupby(["ts", "candidate"], observed=True)["trade_return"]
        .mean()
        .reset_index()
        .pivot(index="ts", columns="candidate", values="trade_return")
        .sort_index()
        .fillna(0.0)
    )
    return by_ts


def permutation_test_mean(
    returns: pd.Series,
    *,
    n_perm: int = 2000,
    seed: int = 11,
) -> dict[str, float]:
    vals = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 5:
        return {"perm_stat": 0.0, "perm_pvalue": 1.0}
    obs = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(vals), replace=True)
        draws.append(float(np.mean(vals * signs)))
    draws_arr = np.asarray(draws, dtype=float)
    pval = float(np.mean(np.abs(draws_arr) >= abs(obs)))
    return {"perm_stat": obs, "perm_pvalue": pval}


def white_reality_check(
    candidate_returns: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 17,
) -> dict[str, float]:
    if candidate_returns.empty or candidate_returns.shape[1] == 0:
        return {"white_stat": 0.0, "white_pvalue": 1.0, "n_candidates": 0.0, "white_n_obs": 0.0}
    if candidate_returns.shape[0] < 8:
        obs = float(np.max(np.mean(candidate_returns.to_numpy(dtype=float), axis=0)))
        return {
            "white_stat": obs,
            "white_pvalue": 1.0,
            "n_candidates": float(candidate_returns.shape[1]),
            "white_n_obs": float(candidate_returns.shape[0]),
        }

    mat = candidate_returns.to_numpy(dtype=float)
    means = np.mean(mat, axis=0)
    obs_stat = float(np.max(means))
    centered = mat - means
    rng = np.random.default_rng(seed)
    boot_stats: list[float] = []
    n = centered.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = centered[idx]
        boot_stats.append(float(np.max(np.mean(sample, axis=0))))
    pval = float(np.mean(np.asarray(boot_stats, dtype=float) >= obs_stat))
    return {
        "white_stat": obs_stat,
        "white_pvalue": pval,
        "n_candidates": float(candidate_returns.shape[1]),
        "white_n_obs": float(candidate_returns.shape[0]),
    }


def probability_backtest_overfitting(
    candidate_returns: pd.DataFrame,
    *,
    n_slices: int = 8,
    max_trials: int = 200,
    seed: int = 23,
) -> dict[str, float]:
    if candidate_returns.empty or candidate_returns.shape[1] < 2 or candidate_returns.shape[0] < 24:
        return {"pbo": 1.0, "pbo_trials": 0.0, "pbo_median_logit": 0.0, "pbo_n_obs": float(candidate_returns.shape[0])}

    frame = candidate_returns.copy().sort_index()
    n_rows = len(frame)
    n_slices = int(np.clip(n_slices, 4, min(12, n_rows // 3)))
    if n_slices < 4:
        return {"pbo": 1.0, "pbo_trials": 0.0, "pbo_median_logit": 0.0, "pbo_n_obs": float(candidate_returns.shape[0])}

    # Slice the timeline and aggregate per slice to stabilize sparse observations.
    blocks = np.array_split(np.arange(n_rows), n_slices)
    block_perf = []
    for idx in blocks:
        block_perf.append(frame.iloc[idx].mean(axis=0))
    perf = pd.DataFrame(block_perf).reset_index(drop=True)
    s = len(perf)
    half = s // 2
    if half < 2:
        return {"pbo": 1.0, "pbo_trials": 0.0, "pbo_median_logit": 0.0, "pbo_n_obs": float(candidate_returns.shape[0])}

    all_indices = list(range(s))
    combos = list(itertools.combinations(all_indices, half))
    if len(combos) > max_trials:
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(combos), size=max_trials, replace=False)
        combos = [combos[int(i)] for i in picks]

    logits: list[float] = []
    for train_idx in combos:
        train_set = set(train_idx)
        test_idx = [i for i in all_indices if i not in train_set]
        if not test_idx:
            continue
        train_perf = perf.iloc[list(train_set)].mean(axis=0)
        best = str(train_perf.idxmax())

        test_perf = perf.iloc[test_idx].mean(axis=0).sort_values(ascending=False)
        if best not in test_perf.index:
            continue
        rank = int(np.where(test_perf.index.to_numpy() == best)[0][0]) + 1
        m = len(test_perf)
        percentile = rank / float(m + 1)
        percentile = float(np.clip(percentile, 1e-4, 1 - 1e-4))
        logit = float(np.log(percentile / (1.0 - percentile)))
        logits.append(logit)

    if not logits:
        return {"pbo": 1.0, "pbo_trials": 0.0, "pbo_median_logit": 0.0, "pbo_n_obs": float(candidate_returns.shape[0])}
    logs = np.asarray(logits, dtype=float)
    pbo = float(np.mean(logs <= 0.0))
    return {
        "pbo": pbo,
        "pbo_trials": float(len(logs)),
        "pbo_median_logit": float(np.median(logs)),
        "pbo_n_obs": float(candidate_returns.shape[0]),
    }


def run_advanced_validation(
    trades: pd.DataFrame,
    *,
    chosen_threshold: float,
    chosen_holding_period: int,
) -> pd.DataFrame:
    matrix = _candidate_return_matrix(trades)
    if matrix.empty:
        return pd.DataFrame(
            [
                {
                    "perm_stat": 0.0,
                    "perm_pvalue": 1.0,
                    "white_stat": 0.0,
                    "white_pvalue": 1.0,
                    "pbo": 1.0,
                    "pbo_trials": 0.0,
                    "pbo_median_logit": 0.0,
                    "white_n_obs": 0.0,
                    "pbo_n_obs": 0.0,
                    "n_candidates": 0.0,
                }
            ]
        )

    chosen_key = _candidate_key(chosen_threshold, chosen_holding_period)
    if chosen_key not in matrix.columns:
        chosen_key = str(matrix.columns[0])
    perm = permutation_test_mean(matrix[chosen_key], n_perm=2000, seed=11)
    white = white_reality_check(matrix, n_boot=1000, seed=17)
    pbo = probability_backtest_overfitting(matrix, n_slices=8, max_trials=200, seed=23)

    row = {}
    row.update(perm)
    row.update(white)
    row.update(pbo)
    row["chosen_candidate"] = chosen_key
    return pd.DataFrame([row])
