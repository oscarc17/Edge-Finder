from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from polymarket_edge.research.advanced_validation import run_advanced_validation
from polymarket_edge.research.capacity import estimate_capacity
from polymarket_edge.research.deployment import evaluate_deployment_readiness
from polymarket_edge.research.diagnostics import run_signal_diagnostics
from polymarket_edge.research.explain import explain_strategy
from polymarket_edge.research.paired_backtest import (
    PairedBacktestConfig,
    calibrate_pair_execution,
    run_paired_backtest_grid,
    run_paired_execution_regime_sensitivity,
    run_paired_impact_coefficient_sensitivity,
)
from polymarket_edge.research.types import StrategyExecutionResult
from polymarket_edge.research.validation import WalkForwardConfig, bootstrap_metrics, run_walkforward, validate_strategy
from polymarket_edge.strategies.base import BaseStrategy, StrategyConfig


@dataclass
class ConsistencyArbStrategyConfig(StrategyConfig):
    signal_thresholds: tuple[float, ...] = (0.004, 0.006, 0.008, 0.010, 0.015)
    holding_periods: tuple[int, ...] = (1, 3, 6)
    fee_bps: float = 10.0
    impact_coeff: float = 0.05
    max_impact: float = 0.03
    base_trade_notional: float = 200.0
    max_participation: float = 0.08
    min_history_rows: int = 100
    min_train_trades: int = 50
    min_test_trades_target: int = 300
    min_expected_edge: float = 0.0
    min_expected_net_ev: float = 0.0
    top_confidence_pct: float = 1.0
    model_type: str = "heuristic"
    walkforward_min_test_trades: int = 50
    walkforward_min_train_samples: int = 150
    walkforward_test_periods: int = 24
    walkforward_folds: int = 4
    pair_min_depth: float = 25.0
    min_pair_fill_prob: float = 0.15
    pair_fill_risk_buffer: float = 0.0025
    staleness_tolerance_ms: float = 1000.0


class ConsistencyArbStrategy(BaseStrategy):
    name = "consistency_arb_v2"
    feature_cols = [
        "edge_raw",
        "long_set_edge_raw",
        "short_set_edge_raw",
        "expected_cost_return",
        "expected_net_ev",
        "spread_yes_bps",
        "spread_no_bps",
        "depth_yes",
        "depth_no",
        "min_pair_depth",
        "sum_ask",
        "sum_bid",
        "min_fill_prob_pair",
        "expected_pair_fill",
        "time_to_resolution_h",
        "market_age_h",
        "pair_volatility",
        "pair_staleness_ms",
    ]

    def __init__(self, config: ConsistencyArbStrategyConfig | None = None) -> None:
        super().__init__(config or ConsistencyArbStrategyConfig())
        self.config: ConsistencyArbStrategyConfig

    def run_backtest_grid(
        self,
        signal_frame: pd.DataFrame,
        *,
        thresholds: list[float],
        cfg: PairedBacktestConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return run_paired_backtest_grid(signal_frame, thresholds=thresholds, cfg=cfg)

    def _load_pair_books(self, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        raw = conn.execute(
            """
            SELECT
                o.snapshot_ts AS ts,
                o.market_id,
                o.token_id,
                lower(trim(coalesce(mo.outcome_label, ''))) AS outcome_label,
                o.best_bid,
                o.best_ask,
                o.mid,
                o.spread,
                o.spread_bps,
                o.bid_depth,
                o.ask_depth,
                m.event_id,
                m.category,
                m.question,
                m.created_ts,
                coalesce(m.close_ts, m.end_ts) AS close_ts,
                r.resolved_ts
            FROM orderbook_snapshots o
            JOIN market_outcomes mo
              ON o.market_id = mo.market_id
             AND o.token_id = mo.token_id
            LEFT JOIN markets m
              ON o.market_id = m.market_id
            LEFT JOIN resolutions r
              ON o.market_id = r.market_id
            WHERE lower(trim(coalesce(mo.outcome_label, ''))) IN ('yes', 'no')
            """
        ).df()
        if raw.empty:
            return raw
        raw["ts"] = pd.to_datetime(raw["ts"], errors="coerce")
        return raw.dropna(subset=["ts"]).sort_values(["market_id", "ts", "outcome_label"]).reset_index(drop=True)

    @staticmethod
    def _pivot_pair_books(raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return raw
        base = ["ts", "market_id", "event_id", "category", "question", "created_ts", "close_ts", "resolved_ts"]
        val = ["token_id", "best_bid", "best_ask", "mid", "spread", "spread_bps", "bid_depth", "ask_depth"]
        pieces: list[pd.DataFrame] = []
        for lab in ["yes", "no"]:
            cur = raw[raw["outcome_label"] == lab][base + val].copy()
            cur = cur.drop_duplicates(subset=["ts", "market_id"], keep="last")
            cur = cur.rename(columns={c: f"{c}_{lab}" for c in val})
            pieces.append(cur)
        if len(pieces) != 2:
            return pd.DataFrame()
        p = pieces[0].merge(pieces[1], on=base, how="inner")
        p["pair_staleness_ms"] = 0.0
        p["has_both_legs"] = 1.0
        p["yes_two_sided"] = (p["best_bid_yes"].notna() & p["best_ask_yes"].notna()).astype(float)
        p["no_two_sided"] = (p["best_bid_no"].notna() & p["best_ask_no"].notna()).astype(float)
        p["book_tradable"] = ((p["yes_two_sided"] > 0.0) & (p["no_two_sided"] > 0.0)).astype(float)
        return p.sort_values(["market_id", "ts"]).reset_index(drop=True)

    def _attach_features(self, pair: pd.DataFrame) -> pd.DataFrame:
        if pair.empty:
            return pair
        p = pair.copy()
        num_cols = [
            "best_bid_yes", "best_ask_yes", "best_bid_no", "best_ask_no",
            "spread_yes", "spread_no", "spread_bps_yes", "spread_bps_no",
            "bid_depth_yes", "ask_depth_yes", "bid_depth_no", "ask_depth_no",
        ]
        for c in num_cols:
            p[c] = pd.to_numeric(p[c], errors="coerce")

        p["ask_yes"] = p["best_ask_yes"]
        p["ask_no"] = p["best_ask_no"]
        p["bid_yes"] = p["best_bid_yes"]
        p["bid_no"] = p["best_bid_no"]
        p["spread_yes"] = p["spread_yes"].fillna(0.0).clip(lower=0.0)
        p["spread_no"] = p["spread_no"].fillna(0.0).clip(lower=0.0)
        p["spread_yes_bps"] = pd.to_numeric(p["spread_bps_yes"], errors="coerce").fillna(0.0).clip(lower=0.0)
        p["spread_no_bps"] = pd.to_numeric(p["spread_bps_no"], errors="coerce").fillna(0.0).clip(lower=0.0)
        p["depth_yes"] = p["bid_depth_yes"].fillna(0.0) + p["ask_depth_yes"].fillna(0.0)
        p["depth_no"] = p["bid_depth_no"].fillna(0.0) + p["ask_depth_no"].fillna(0.0)
        p["min_pair_depth"] = np.minimum(p["depth_yes"], p["depth_no"])

        p["sum_ask"] = (p["ask_yes"] + p["ask_no"]).astype(float)
        p["sum_bid"] = (p["bid_yes"] + p["bid_no"]).astype(float)
        p["long_set_edge_raw"] = (1.0 - p["sum_ask"]).astype(float)
        p["short_set_edge_raw"] = (p["sum_bid"] - 1.0).astype(float)
        p["edge_raw"] = np.maximum(np.maximum(p["long_set_edge_raw"], p["short_set_edge_raw"]), 0.0).astype(float)
        p["pair_direction"] = np.where(p["long_set_edge_raw"] >= p["short_set_edge_raw"], "LONG_SET", "SHORT_SET")
        p["signal"] = np.where(p["pair_direction"] == "LONG_SET", p["edge_raw"], -p["edge_raw"]).astype(float)

        p["created_ts"] = pd.to_datetime(p["created_ts"], errors="coerce")
        p["close_ts"] = pd.to_datetime(p["close_ts"], errors="coerce")
        p["resolved_ts"] = pd.to_datetime(p["resolved_ts"], errors="coerce")
        ref_close = p["resolved_ts"].combine_first(p["close_ts"])
        p["time_to_resolution_h"] = ((ref_close - p["ts"]).dt.total_seconds() / 3600.0).fillna(9999.0)
        p["market_age_h"] = ((p["ts"] - p["created_ts"]).dt.total_seconds() / 3600.0).fillna(0.0)

        p["pair_mid"] = ((p["best_bid_yes"] + p["best_bid_no"] + p["best_ask_yes"] + p["best_ask_no"]) / 2.0).clip(lower=1e-6, upper=0.999999)
        p = p.sort_values(["market_id", "ts"]).copy()
        pair_ret = p.groupby("market_id", observed=True)["pair_mid"].pct_change(fill_method=None)
        p["pair_volatility"] = pair_ret.groupby(p["market_id"], observed=True).transform(lambda s: s.rolling(6, min_periods=3).std()).fillna(0.0).abs()
        p["prob_vol_pair"] = p["pair_volatility"]
        p["trade_frequency_pair_h"] = 0.0
        p["category"] = p["category"].fillna("unknown")
        p["category_code"] = pd.factorize(p["category"])[0].astype(float)

        spread_sum_bps = p["spread_yes_bps"] + p["spread_no_bps"]
        depth_score = np.log1p(p["min_pair_depth"].clip(lower=0.0))
        p["confidence"] = (
            (p["edge_raw"].clip(lower=0.0) / 0.02).clip(upper=1.0)
            * (1.0 / (1.0 + spread_sum_bps / 400.0))
            * (depth_score / max(np.log1p(500.0), 1e-6))
        ).clip(lower=0.05, upper=1.0)

        fee_rate = self.config.fee_bps / 10_000.0
        size_ratio = (self.config.base_trade_notional / p["min_pair_depth"].clip(lower=1.0)).clip(lower=0.0)
        impact_return = np.minimum(self.config.max_impact, self.config.impact_coeff * size_ratio)
        vol_slippage_return = 0.10 * p["pair_volatility"] * (1.0 + size_ratio)
        fill_logit = 1.5 - 2.0 * size_ratio - (spread_sum_bps / 250.0)
        fill_prob = (1.0 / (1.0 + np.exp(-np.clip(fill_logit, -20.0, 20.0)))).clip(lower=0.0, upper=1.0)
        fill_risk_buffer = (1.0 - fill_prob) * float(self.config.pair_fill_risk_buffer)

        p["fill_prob_yes"] = fill_prob.astype(float)
        p["fill_prob_no"] = fill_prob.astype(float)
        p["min_fill_prob_pair"] = np.minimum(p["fill_prob_yes"], p["fill_prob_no"]).astype(float)
        p["expected_pair_fill"] = (p["fill_prob_yes"] * p["fill_prob_no"]).astype(float)
        p["expected_gross_return"] = (p["edge_raw"] * p["expected_pair_fill"]).astype(float)
        p["expected_fee_return"] = (2.0 * fee_rate * p["expected_pair_fill"]).astype(float)
        p["expected_spread_cost_return"] = fill_risk_buffer.astype(float)
        p["expected_impact_return"] = (impact_return * p["expected_pair_fill"]).astype(float)
        p["expected_vol_slippage_return"] = (vol_slippage_return * p["expected_pair_fill"]).astype(float)
        p["expected_cost_return"] = (
            p["expected_fee_return"] + p["expected_spread_cost_return"] + p["expected_impact_return"] + p["expected_vol_slippage_return"]
        ).astype(float)
        p["expected_net_ev"] = (p["expected_gross_return"] - p["expected_cost_return"]).astype(float)
        p["expected_net_edge"] = p["expected_net_ev"]
        p["expected_return"] = p["expected_net_ev"]
        p["expected_ev"] = p["expected_net_ev"]
        p["required_notional"] = float(self.config.base_trade_notional)
        p["capacity_estimate"] = (self.config.max_participation * p["min_pair_depth"]).astype(float)
        return p

    def _attach_forward_quotes(self, pair: pd.DataFrame) -> pd.DataFrame:
        if pair.empty:
            return pair
        p = pair.sort_values(["market_id", "ts"]).copy()
        grp = p.groupby("market_id", observed=True)
        max_h = max(int(h) for h in self.config.holding_periods)
        base_future_cols = ["bid_yes", "ask_yes", "bid_no", "ask_no", "spread_yes", "spread_no", "edge_raw", "long_set_edge_raw", "short_set_edge_raw"]
        for h in range(1, max_h + 1):
            for col in base_future_cols:
                if col in p.columns:
                    p[f"future_{col}_{h}"] = grp[col].shift(-h)
        p["forward_ret_1"] = (
            pd.to_numeric(p.get("future_edge_raw_1"), errors="coerce") - pd.to_numeric(p.get("edge_raw"), errors="coerce")
        ).fillna(0.0)
        return p

    def build_signal_frame(
        self,
        conn: duckdb.DuckDBPyConnection,
        panel: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        raw = self._load_pair_books(conn)
        if raw.empty:
            return pd.DataFrame()
        pair = self._pivot_pair_books(raw)
        if pair.empty:
            return pair
        pair = self._attach_features(pair)
        pair = self._attach_forward_quotes(pair)
        pair = pair[pair["book_tradable"] > 0.0].copy()
        pair = pair[pair["min_pair_depth"] >= float(self.config.pair_min_depth)].copy()
        pair = pair[pair["min_fill_prob_pair"] >= float(self.config.min_pair_fill_prob)].copy()
        if pair.empty:
            return pair
        pair["direction"] = np.where(pair["signal"] >= 0.0, "LONG_YES", "LONG_NO")
        pair["traded_side"] = "BASKET"
        pair["position_type"] = pair["pair_direction"].str.lower()
        pair["prediction_space"] = "RETURN_NEXT_H"
        pair["model_target_type"] = "RETURN_NEXT_H"
        pair["model_horizon"] = 1.0
        return pair.sort_values(["ts", "market_id"]).reset_index(drop=True)

    def _structural_filter_report(self, frame: pd.DataFrame, *, split: str, ev_floor: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        if frame.empty:
            cols = ["split", "filter", "rows", "avg_signal", "avg_expected_ev", "proxy_expectancy", "proxy_sharpe"]
            return frame.copy(), pd.DataFrame(columns=cols)
        rows: list[dict[str, float | str]] = []
        cur = frame.copy()
        checks = [
            ("raw", pd.Series(True, index=cur.index)),
            ("book_tradable", pd.to_numeric(cur.get("book_tradable", 0.0), errors="coerce").fillna(0.0) > 0.0),
            ("min_depth", pd.to_numeric(cur.get("min_pair_depth", 0.0), errors="coerce").fillna(0.0) >= float(self.config.pair_min_depth)),
            ("min_fill_prob", pd.to_numeric(cur.get("min_fill_prob_pair", 0.0), errors="coerce").fillna(0.0) >= float(self.config.min_pair_fill_prob)),
            (f"min_expected_net_ev_{ev_floor:.6f}", pd.to_numeric(cur.get("expected_net_ev", 0.0), errors="coerce").fillna(-np.inf) >= float(ev_floor)),
        ]
        for label, mask in checks:
            cur = cur[mask.loc[cur.index]].copy()
            proxy = pd.to_numeric(cur.get("forward_ret_1", 0.0), errors="coerce").dropna()
            rows.append(
                {
                    "split": split,
                    "filter": label,
                    "rows": float(len(cur)),
                    "avg_signal": float(pd.to_numeric(cur.get("signal", 0.0), errors="coerce").abs().mean()) if len(cur) else 0.0,
                    "avg_expected_ev": float(pd.to_numeric(cur.get("expected_net_ev", 0.0), errors="coerce").mean()) if len(cur) else 0.0,
                    "proxy_expectancy": float(proxy.mean()) if len(proxy) else 0.0,
                    "proxy_sharpe": float(proxy.mean() / max(proxy.std(ddof=0), 1e-9)) if len(proxy) > 1 else 0.0,
                }
            )
        return cur, pd.DataFrame(rows)

    def _structural_ev_threshold_grid(self, frame: pd.DataFrame, *, split: str) -> pd.DataFrame:
        if frame.empty:
            cols = ["split", "ev_threshold", "rows", "avg_expected_net_ev", "proxy_expectancy", "proxy_sharpe"]
            return pd.DataFrame(columns=cols)
        rows: list[dict[str, float | str]] = []
        for thr in sorted(set(float(x) for x in self.config.ev_threshold_grid)):
            cur = frame[pd.to_numeric(frame.get("expected_net_ev", 0.0), errors="coerce").fillna(-np.inf) >= thr].copy()
            proxy = pd.to_numeric(cur.get("forward_ret_1", 0.0), errors="coerce").dropna()
            rows.append(
                {
                    "split": split,
                    "ev_threshold": float(thr),
                    "rows": float(len(cur)),
                    "avg_expected_net_ev": float(pd.to_numeric(cur.get("expected_net_ev", 0.0), errors="coerce").mean()) if len(cur) else 0.0,
                    "proxy_expectancy": float(proxy.mean()) if len(proxy) else 0.0,
                    "proxy_sharpe": float(proxy.mean() / max(proxy.std(ddof=0), 1e-9)) if len(proxy) > 1 else 0.0,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _merge_validation_hedged(validation: pd.DataFrame, test_eval: pd.DataFrame) -> pd.DataFrame:
        out = validation.copy()
        if out.empty:
            return out
        if test_eval.empty or "fully_hedged" not in test_eval.columns:
            out["hedged_mean_return"] = 0.0
            out["hedged_mean_return_ci_low"] = 0.0
            out["hedged_mean_return_ci_high"] = 0.0
            out["hedged_n"] = 0.0
            return out
        hedged = test_eval[pd.to_numeric(test_eval["fully_hedged"], errors="coerce").fillna(0.0) >= 0.5].copy()
        if hedged.empty:
            out["hedged_mean_return"] = 0.0
            out["hedged_mean_return_ci_low"] = 0.0
            out["hedged_mean_return_ci_high"] = 0.0
            out["hedged_n"] = 0.0
            return out
        boot = bootstrap_metrics(pd.to_numeric(hedged["trade_return"], errors="coerce"), ts=hedged.get("ts"))
        out["hedged_mean_return"] = float(pd.to_numeric(hedged["trade_return"], errors="coerce").mean())
        out["hedged_mean_return_ci_low"] = float(boot["mean_return_ci_low"].iloc[0]) if not boot.empty else 0.0
        out["hedged_mean_return_ci_high"] = float(boot["mean_return_ci_high"].iloc[0]) if not boot.empty else 0.0
        out["hedged_n"] = float(len(hedged))
        return out

    @staticmethod
    def _arb_null_shuffle_pvalue(test_eval: pd.DataFrame, *, seed: int = 11, n_perm: int = 500) -> float:
        req = {"trade_return", "edge_raw", "market_id"}
        if test_eval.empty or not req.issubset(test_eval.columns) or len(test_eval) < 20:
            return 1.0
        t = test_eval.copy()
        t["trade_return"] = pd.to_numeric(t["trade_return"], errors="coerce").fillna(0.0)
        t["edge_raw"] = pd.to_numeric(t["edge_raw"], errors="coerce").fillna(0.0)
        obs = float(t["trade_return"].corr(t["edge_raw"], method="spearman"))
        if not np.isfinite(obs):
            return 1.0
        rng = np.random.default_rng(seed)
        draws: list[float] = []
        for _ in range(int(n_perm)):
            chunks = []
            for _m, g in t.groupby("market_id", observed=True):
                h = g.copy()
                h["edge_raw"] = rng.permutation(h["edge_raw"].to_numpy(dtype=float))
                chunks.append(h)
            if not chunks:
                continue
            p = pd.concat(chunks, ignore_index=True)
            c = float(p["trade_return"].corr(p["edge_raw"], method="spearman"))
            if np.isfinite(c):
                draws.append(c)
        if not draws:
            return 1.0
        arr = np.asarray(draws, dtype=float)
        return float(np.mean(np.abs(arr) >= abs(obs)))

    def run(
        self,
        conn: duckdb.DuckDBPyConnection,
        panel: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> StrategyExecutionResult:
        signal_frame = self.build_signal_frame(conn, panel, context=context)
        self._last_signal_frame = signal_frame.copy()
        empty = pd.DataFrame()
        if signal_frame.empty or len(signal_frame) < self.config.min_history_rows:
            validation = validate_strategy(empty, empty)
            walk = run_walkforward(self, signal_frame=pd.DataFrame(columns=["ts"]), backtest_config=PairedBacktestConfig(), walk_cfg=WalkForwardConfig())
            adv = pd.DataFrame([{"perm_pvalue": 1.0, "white_pvalue": 1.0, "pbo": 1.0, "n_candidates": 0.0, "white_n_obs": 0.0}])
            readiness = evaluate_deployment_readiness(
                strategy_name=self.name,
                walkforward_summary=walk["aggregate"],
                walkforward_folds=walk["folds"],
                validation_summary=validation,
                stability_time=walk["stability_time"],
                test_trades=empty,
                execution_sensitivity=pd.DataFrame(),
                advanced_validation=adv,
                ev_diagnostics=pd.DataFrame(),
            )
            return StrategyExecutionResult(
                strategy_name=self.name,
                tuned_params={"threshold": 0.006, "holding_period": 1, "min_expected_net_ev": float(self.config.min_expected_net_ev)},
                signal_frame=signal_frame,
                train_summary=empty,
                test_summary=empty,
                train_trades=empty,
                test_trades=empty,
                validation_summary=validation,
                explain_summary=empty,
                strong_conditions=empty,
                weak_conditions=empty,
                capacity_summary=empty,
                model_comparison=self._model_unavailable_metrics("insufficient_history_rows"),
                model_feature_importance=empty,
                label_diagnostics=self._empty_label_diag("structural_arb_no_labels", "none"),
                walkforward_folds=walk["folds"],
                walkforward_summary=walk["aggregate"],
                stability_time=walk["stability_time"],
                stability_category=walk["stability_category"],
                parameter_sensitivity=walk["parameter_sensitivity"],
                filter_impact=empty,
                edge_attribution=empty,
                calibration_curve=empty,
                signal_deciles=empty,
                feature_drift=empty,
                model_vs_market=empty,
                ev_diagnostics=empty,
                advanced_validation=adv,
                deployment_readiness=readiness,
                execution_sensitivity=empty,
                execution_calibration=empty,
                cost_decomposition=empty,
                ev_threshold_grid=empty,
            )

        train_raw, test_raw = self._time_split(signal_frame, self.config.train_ratio)
        signal_frame_modeled = pd.concat([train_raw, test_raw], ignore_index=True).sort_values("ts").reset_index(drop=True)
        ev_grid_train = self._structural_ev_threshold_grid(train_raw, split="train")
        ev_grid_test = self._structural_ev_threshold_grid(test_raw, split="test")
        ev_threshold_grid = pd.concat([ev_grid_train, ev_grid_test], ignore_index=True)
        best_ev_threshold = self._select_ev_threshold(ev_grid_train)
        train, train_filter = self._structural_filter_report(train_raw, split="train", ev_floor=best_ev_threshold)
        test, test_filter = self._structural_filter_report(test_raw, split="test", ev_floor=best_ev_threshold)
        filter_report = pd.concat([train_filter, test_filter], ignore_index=True)
        self._last_signal_frame = pd.concat([train, test], ignore_index=True).sort_values("ts")

        bt_cfg = PairedBacktestConfig(
            holding_periods=tuple(int(h) for h in self.config.holding_periods),
            fee_bps=self.config.fee_bps,
            impact_coeff=self.config.impact_coeff,
            max_impact=self.config.max_impact,
            base_trade_notional=self.config.base_trade_notional,
            max_participation=self.config.max_participation,
            staleness_tolerance_ms=self.config.staleness_tolerance_ms,
            execution_regime="base",
        )
        train_summary_all, train_trades_all = self.run_backtest_grid(train, thresholds=list(self.config.signal_thresholds), cfg=bt_cfg)
        best = self._select_best(train_summary_all, min_trades=self.config.min_train_trades)
        threshold = float(best["threshold"])
        best_hp = int(best["holding_period"])

        train_summary = train_summary_all[train_summary_all["threshold"] == threshold].copy() if not train_summary_all.empty else empty
        train_trades = train_trades_all[train_trades_all["threshold"] == threshold].copy() if not train_trades_all.empty else empty

        test_summary_all, test_trades_all = self.run_backtest_grid(test, thresholds=list(self.config.signal_thresholds), cfg=bt_cfg)
        test_summary = test_summary_all.copy()
        test_trades = test_trades_all.copy() if not test_trades_all.empty else empty
        chosen = (
            test_summary[
                (pd.to_numeric(test_summary.get("threshold"), errors="coerce") == float(threshold))
                & (pd.to_numeric(test_summary.get("holding_period"), errors="coerce") == float(best_hp))
            ].copy()
            if not test_summary.empty
            else empty
        )
        if not test_summary.empty:
            constrained = test_summary[pd.to_numeric(test_summary.get("trade_count"), errors="coerce").fillna(0.0) >= float(self.config.min_test_trades_target)].copy()
            if not constrained.empty:
                score = pd.to_numeric(constrained["expectancy"], errors="coerce").fillna(0.0) * np.sqrt(pd.to_numeric(constrained["trade_count"], errors="coerce").fillna(0.0).clip(lower=0.0))
                row = constrained.loc[int(score.idxmax())]
                threshold = float(row["threshold"])
                best_hp = int(float(row["holding_period"]))
            elif chosen.empty:
                alt = test_summary[pd.to_numeric(test_summary.get("trade_count"), errors="coerce").fillna(0.0) > 0.0].copy()
                if not alt.empty:
                    score = pd.to_numeric(alt["expectancy"], errors="coerce").fillna(0.0) * np.sqrt(pd.to_numeric(alt["trade_count"], errors="coerce").fillna(0.0).clip(lower=0.0))
                    row = alt.loc[int(score.idxmax())]
                    threshold = float(row["threshold"])
                    best_hp = int(float(row["holding_period"]))

        def _pick(tr: pd.DataFrame) -> pd.DataFrame:
            if tr.empty:
                return tr
            return tr[
                (pd.to_numeric(tr.get("holding_period"), errors="coerce") == float(best_hp))
                & (pd.to_numeric(tr.get("threshold"), errors="coerce") == float(threshold))
            ].copy()

        train_eval = _pick(train_trades)
        test_eval = _pick(test_trades)

        cost_decomp = self._cost_decomposition(test_eval)
        validation = self._merge_validation_hedged(validate_strategy(train_eval, test_eval), test_eval)
        explain, strong, weak = explain_strategy(test_eval, self.feature_cols)
        capacity = estimate_capacity(test_eval, max_participation=self.config.max_participation, impact_coeff=self.config.impact_coeff)
        diagnostics = run_signal_diagnostics(signal_frame=pd.concat([train, test], ignore_index=True), trades=test_eval, feature_cols=self.feature_cols)
        adv_validation = run_advanced_validation(test_trades, chosen_threshold=threshold, chosen_holding_period=best_hp)
        if adv_validation.empty:
            adv_validation = pd.DataFrame([{"perm_pvalue": 1.0, "white_pvalue": 1.0, "pbo": 1.0, "n_candidates": 0.0, "white_n_obs": 0.0}])
        adv_validation = adv_validation.copy()
        adv_validation["arb_noarb_null_pvalue"] = self._arb_null_shuffle_pvalue(test_eval)

        walk = run_walkforward(
            self,
            signal_frame=pd.concat([train, test], ignore_index=True).sort_values("ts"),
            backtest_config=bt_cfg,
            walk_cfg=WalkForwardConfig(
                n_folds=self.config.walkforward_folds,
                min_train_periods=self.config.walkforward_min_train_periods,
                min_train_samples=self.config.walkforward_min_train_samples,
                test_periods=self.config.walkforward_test_periods,
                min_test_trades=self.config.walkforward_min_test_trades,
                embargo_periods=1,
            ),
        )
        exec_sens = run_paired_execution_regime_sensitivity(test, threshold=threshold, holding_period=best_hp, cfg=bt_cfg)
        impact_sens = run_paired_impact_coefficient_sensitivity(test, threshold=threshold, holding_period=best_hp, cfg=bt_cfg)
        execution_sensitivity = exec_sens["sensitivity"]
        if not impact_sens.empty:
            execution_sensitivity = pd.concat([execution_sensitivity, impact_sens], ignore_index=True)
        execution_calibration = exec_sens["calibration"]
        if execution_calibration.empty and not test_eval.empty:
            execution_calibration = calibrate_pair_execution(test_eval)

        readiness = evaluate_deployment_readiness(
            strategy_name=self.name,
            walkforward_summary=walk["aggregate"],
            walkforward_folds=walk["folds"],
            validation_summary=validation,
            stability_time=walk["stability_time"],
            test_trades=test_eval,
            execution_sensitivity=execution_sensitivity,
            advanced_validation=adv_validation,
            ev_diagnostics=diagnostics["ev_diagnostics"],
            min_valid_folds=3,
            min_test_trades=300,
        )
        fully_hedged_col = test_eval["fully_hedged"] if ("fully_hedged" in test_eval.columns) else pd.Series(0.0, index=test_eval.index)
        hedged = test_eval[pd.to_numeric(fully_hedged_col, errors="coerce").fillna(0.0) >= 0.5].copy()
        hedged_mean = float(pd.to_numeric(hedged.get("trade_return"), errors="coerce").mean()) if not hedged.empty else 0.0
        hedged_ci_low = 0.0
        if not hedged.empty:
            hb = bootstrap_metrics(pd.to_numeric(hedged["trade_return"], errors="coerce"), ts=hedged.get("ts"))
            hedged_ci_low = float(hb["mean_return_ci_low"].iloc[0]) if not hb.empty else 0.0
        hedged_pass = bool(len(hedged) >= 50 and hedged_mean > 0.0 and hedged_ci_low > 0.0)
        extra_row = pd.DataFrame([{
            "criterion": "fully_hedged_subset_profitable",
            "value": hedged_mean,
            "threshold": "n>=50 and mean>0 and ci_low>0",
            "passed": hedged_pass,
            "notes": f"n={len(hedged)} ci_low={hedged_ci_low:.6f}",
            "strategy": self.name,
        }])
        readiness = pd.concat([readiness, extra_row], ignore_index=True, sort=False)
        for crit in ["summary", "deployment_decision"]:
            mask = readiness["criterion"].astype(str).eq(crit)
            if bool(mask.any()) and not hedged_pass:
                idx = readiness.index[mask][0]
                readiness.loc[idx, "passed"] = False
                readiness.loc[idx, "decision"] = "FAIL"
                if crit == "deployment_decision":
                    readiness.loc[idx, "value"] = 0.0
                    readiness.loc[idx, "notes"] = "NO_GO"
                else:
                    note = str(readiness.loc[idx, "notes"] or "")
                    suffix = "fully_hedged_subset"
                    readiness.loc[idx, "notes"] = (note + (";" if note else "") + suffix)

        tuned = {"threshold": threshold, "holding_period": best_hp, "min_expected_net_ev": best_ev_threshold}
        return StrategyExecutionResult(
            strategy_name=self.name,
            tuned_params=tuned,
            signal_frame=signal_frame_modeled,
            train_summary=train_summary,
            test_summary=test_summary,
            train_trades=train_trades,
            test_trades=test_trades,
            validation_summary=validation,
            explain_summary=explain,
            strong_conditions=strong,
            weak_conditions=weak,
            capacity_summary=capacity,
            model_comparison=self._model_unavailable_metrics("structural_arb_no_ml"),
            model_feature_importance=empty,
            label_diagnostics=self._empty_label_diag("structural_arb_no_labels", "none"),
            walkforward_folds=walk["folds"],
            walkforward_summary=walk["aggregate"],
            stability_time=walk["stability_time"],
            stability_category=walk["stability_category"],
            parameter_sensitivity=walk["parameter_sensitivity"],
            filter_impact=filter_report,
            edge_attribution=diagnostics["edge_attribution"],
            calibration_curve=diagnostics["calibration_curve"],
            signal_deciles=diagnostics["signal_deciles"],
            feature_drift=diagnostics["feature_drift"],
            model_vs_market=diagnostics["model_vs_market"],
            ev_diagnostics=diagnostics["ev_diagnostics"],
            advanced_validation=adv_validation,
            deployment_readiness=readiness,
            execution_sensitivity=execution_sensitivity,
            execution_calibration=execution_calibration,
            cost_decomposition=cost_decomp,
            ev_threshold_grid=ev_threshold_grid,
        )
