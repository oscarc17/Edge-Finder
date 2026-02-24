from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from polymarket_edge.backtest.metrics import sharpe_ratio
from polymarket_edge.models import compare_models, create_alpha_model, fit_predict_future_mid_ensemble
from polymarket_edge.research.advanced_validation import run_advanced_validation
from polymarket_edge.research.backtest import (
    StrategyBacktestConfig,
    run_backtest_grid,
    run_impact_coefficient_sensitivity,
    run_execution_regime_sensitivity,
)
from polymarket_edge.research.capacity import estimate_capacity
from polymarket_edge.research.deployment import evaluate_deployment_readiness
from polymarket_edge.research.direction import (
    DIRECTION_LONG_NO,
    DIRECTION_LONG_YES,
    direction_label_series,
    direction_sign_series,
)
from polymarket_edge.research.diagnostics import run_signal_diagnostics
from polymarket_edge.research.ev_engine import EVEngineConfig, apply_ev_engine
from polymarket_edge.research.explain import explain_strategy
from polymarket_edge.research.semantics import PredictionSpace
from polymarket_edge.research.types import StrategyExecutionResult
from polymarket_edge.research.validation import WalkForwardConfig, run_walkforward, validate_strategy


@dataclass
class StrategyConfig:
    signal_thresholds: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25)
    holding_periods: tuple[int, ...] = (1, 3, 6, 12)
    train_ratio: float = 0.70
    fee_bps: float = 20.0
    impact_coeff: float = 0.10
    max_impact: float = 0.05
    base_trade_notional: float = 250.0
    max_participation: float = 0.15
    min_history_rows: int = 60
    min_train_trades: int = 20
    min_test_trades_target: int = 200
    model_type: str = "heuristic"  # heuristic | logistic | gradient_boosting | bayesian | auto | ensemble
    model_candidates: tuple[str, ...] = ("logistic", "gradient_boosting", "bayesian")
    model_target_horizon: int = 1
    model_cv_splits: int = 4
    ensemble_blend: float = 0.5
    walkforward_folds: int = 4
    walkforward_min_train_periods: int = 4
    walkforward_min_train_samples: int = 120
    walkforward_test_periods: int = 12
    walkforward_min_test_trades: int = 20
    top_confidence_pct: float = 1.00
    min_expected_edge: float = 0.0
    min_expected_net_ev: float = 0.0
    ev_threshold_grid: tuple[float, ...] = (0.0, 0.001, 0.002, 0.005)
    min_liquidity_log: float | None = None
    max_prob_vol_6: float | None = None
    max_spread_bps: float | None = 2_500.0
    allowed_vol_regimes: tuple[int, ...] = (0, 1, 2)
    label_mode: str = "auto"  # auto | future_price_move | resolved_outcome
    label_move_quantiles: tuple[float, ...] = (0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0)
    label_target_pos_rate: float = 0.35
    label_min_pos_rate: float = 0.10
    label_max_pos_rate: float = 0.90
    label_min_samples: int = 200


class BaseStrategy:
    name: str = "base"
    feature_cols: list[str] = [
        "time_to_resolution_h",
        "prob_vol_6",
        "prob_vol_24",
        "depth_total",
        "spread_bps",
        "book_imbalance",
        "category_code",
        "market_age_h",
        "velocity_1",
        "velocity_3",
        "acceleration",
        "liquidity_log",
        "order_flow_imbalance_h",
        "trade_frequency_h",
        "vol_regime",
        "ttr_log",
        "ttr_sq",
        "near_resolution_24h",
        "near_resolution_72h",
        "momentum_decay",
        "related_prob_spread",
        "event_prob_dispersion",
        "category_prob_spread",
        "corr_cluster",
        "token_corr_category",
    ]

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    @staticmethod
    def _empty_label_diag(reason: str, mode: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "label_mode": mode,
                    "label_source": "none",
                    "label_warning": reason,
                    "n_samples": 0.0,
                    "y_unique": 0.0,
                    "pos_rate": 0.0,
                    "move_threshold": 0.0,
                }
            ]
        )

    def _labels_future_price_move(self, frame: pd.DataFrame, *, horizon: int) -> tuple[pd.Series, dict[str, float | str]]:
        target_col = f"future_mid_{horizon}"
        if target_col not in frame.columns:
            return pd.Series(dtype=int), {
                "label_mode": "future_price_move",
                "label_source": target_col,
                "label_warning": f"missing_target:{target_col}",
                "n_samples": 0.0,
                "y_unique": 0.0,
                "pos_rate": 0.0,
                "move_threshold": 0.0,
            }
        mid = pd.to_numeric(frame.get("mid"), errors="coerce")
        future = pd.to_numeric(frame.get(target_col), errors="coerce")
        delta = future - mid
        valid = delta.notna() & mid.notna()
        if int(valid.sum()) < 2:
            return pd.Series(dtype=int), {
                "label_mode": "future_price_move",
                "label_source": target_col,
                "label_warning": "insufficient_valid_moves",
                "n_samples": float(valid.sum()),
                "y_unique": 0.0,
                "pos_rate": 0.0,
                "move_threshold": 0.0,
            }

        d = delta.loc[valid].astype(float)
        target_pos = float(np.clip(self.config.label_target_pos_rate, 0.01, 0.99))
        min_pos = float(np.clip(self.config.label_min_pos_rate, 0.0, 1.0))
        max_pos = float(np.clip(self.config.label_max_pos_rate, 0.0, 1.0))
        q_candidates = [float(np.clip(q, 0.0, 1.0)) for q in self.config.label_move_quantiles]
        q_candidates.append(float(np.clip(1.0 - target_pos, 0.0, 1.0)))

        candidates: list[tuple[float, bool, int, float, pd.Series, float, int]] = []
        for q in q_candidates:
            thr = float(pd.to_numeric(d.quantile(q), errors="coerce"))
            if not np.isfinite(thr):
                continue
            y = (d > thr).astype(int)
            pos = float(y.mean()) if len(y) else 0.0
            uniq = int(y.nunique())
            score = abs(pos - target_pos)
            in_band = (pos >= min_pos) and (pos <= max_pos)
            candidates.append((score, not in_band, -uniq, thr, y, pos, uniq))

        # Tie-heavy series can collapse class balance with simple quantiles.
        # Always include a percentile-rank candidate to target a stable positive-rate.
        ranks = d.rank(method="first", pct=True)
        cutoff = float(np.clip(1.0 - target_pos, 0.0, 1.0))
        y_rank = (ranks >= cutoff).astype(int)
        pos_rank = float(y_rank.mean()) if len(y_rank) else 0.0
        uniq_rank = int(y_rank.nunique())
        thr_rank = float(pd.to_numeric(d.quantile(cutoff), errors="coerce"))
        candidates.append((abs(pos_rank - target_pos), not (min_pos <= pos_rank <= max_pos), -uniq_rank, thr_rank, y_rank, pos_rank, uniq_rank))

        candidates.sort(key=lambda x: (x[1], x[0], x[2]))
        _, _, _, best_thr, best_y, best_pos, best_uniq = candidates[0]
        warn = ""
        if best_uniq < 2:
            warn = "degenerate_train_labels"
        elif not (min_pos <= best_pos <= max_pos):
            warn = "label_imbalance"
        return best_y, {
            "label_mode": "future_price_move",
            "label_source": target_col,
            "label_warning": warn,
            "n_samples": float(len(best_y)),
            "y_unique": float(best_uniq),
            "pos_rate": float(best_pos),
            "move_threshold": float(best_thr),
        }

    def _labels_resolved_outcome(self, frame: pd.DataFrame) -> tuple[pd.Series, dict[str, float | str]]:
        if "winner" not in frame.columns:
            return pd.Series(dtype=int), {
                "label_mode": "resolved_outcome",
                "label_source": "winner",
                "label_warning": "missing_winner_column",
                "n_samples": 0.0,
                "y_unique": 0.0,
                "pos_rate": 0.0,
                "move_threshold": 0.0,
            }
        y_raw = pd.to_numeric(frame["winner"], errors="coerce")
        y = y_raw.dropna().astype(int)
        uniq = int(y.nunique())
        pos = float(y.mean()) if len(y) else 0.0
        warn = ""
        if uniq < 2:
            warn = "degenerate_train_labels"
        return y, {
            "label_mode": "resolved_outcome",
            "label_source": "winner",
            "label_warning": warn,
            "n_samples": float(len(y)),
            "y_unique": float(uniq),
            "pos_rate": float(pos),
            "move_threshold": 0.0,
        }

    def _build_training_labels(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.Series, pd.DataFrame]:
        horizon = int(self.config.model_target_horizon)
        mode = str(self.config.label_mode).lower().strip()
        rows: list[dict[str, float | str]] = []
        if mode in {"resolved_outcome"}:
            y, diag = self._labels_resolved_outcome(frame)
            rows.append(diag)
            return y, pd.DataFrame(rows)
        if mode in {"future_price_move"}:
            y, diag = self._labels_future_price_move(frame, horizon=horizon)
            rows.append(diag)
            return y, pd.DataFrame(rows)

        # auto mode: prefer resolved labels when they exist and are non-degenerate.
        y_res, d_res = self._labels_resolved_outcome(frame)
        rows.append(d_res)
        if int(d_res.get("y_unique", 0.0)) >= 2 and int(d_res.get("n_samples", 0.0)) >= int(self.config.label_min_samples):
            d_res["label_warning"] = str(d_res.get("label_warning", "")) or "ok"
            return y_res, pd.DataFrame(rows)

        y_move, d_move = self._labels_future_price_move(frame, horizon=horizon)
        rows.append(d_move)
        return y_move, pd.DataFrame(rows)

    def _model_feature_set(self, frame: pd.DataFrame) -> list[str]:
        features = [c for c in self.feature_cols if c in frame.columns]
        return features

    def _model_unavailable_metrics(self, reason: str) -> pd.DataFrame:
        names = list(self.config.model_candidates)
        if not names:
            names = ["logistic", "gradient_boosting", "bayesian"]
        rows = []
        for name in names:
            rows.append(
                {
                    "model": name,
                    "auc": 0.5,
                    "log_loss": 0.693,
                    "brier": 0.25,
                    "folds": 0.0,
                    "selected": False,
                    "usable": False,
                    "reason": reason,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _attach_move_size_estimates(
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        target_col: str,
        horizon: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = train.copy()
        test = test.copy()
        move_train = (
            pd.to_numeric(train.get(target_col), errors="coerce")
            - pd.to_numeric(train.get("mid"), errors="coerce")
        ).abs()
        move_train = pd.to_numeric(move_train, errors="coerce")
        global_move = float(move_train.dropna().mean()) if move_train.notna().sum() else 0.01
        global_move = float(np.clip(global_move, 1e-4, 0.25))

        def _assign(frame: pd.DataFrame) -> pd.DataFrame:
            est = pd.Series(global_move, index=frame.index, dtype=float)
            if "category" in frame.columns and "category" in train.columns:
                cat_map = (
                    pd.DataFrame(
                        {
                            "category": train["category"],
                            "m": move_train,
                        }
                    )
                    .dropna()
                    .groupby("category", observed=True)["m"]
                    .mean()
                )
                est = pd.to_numeric(frame["category"].map(cat_map), errors="coerce").fillna(est)
            est = est.clip(lower=1e-4, upper=0.40)
            frame["expected_move_size"] = est.astype(float)
            frame[f"expected_move_size_{horizon}"] = est.astype(float)
            return frame

        return _assign(train), _assign(test)

    def _apply_model_signals(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        key = self.config.model_type.lower().strip()
        if key == "heuristic":
            return train, test, self._model_unavailable_metrics("model_type_heuristic"), pd.DataFrame(), self._empty_label_diag("model_type_heuristic", self.config.label_mode)

        horizon = int(self.config.model_target_horizon)
        target_col = f"future_mid_{horizon}"
        if target_col not in train.columns:
            return train, test, self._model_unavailable_metrics(f"missing_target:{target_col}"), pd.DataFrame(), self._empty_label_diag(f"missing_target:{target_col}", self.config.label_mode)
        features = self._model_feature_set(train)
        if not features:
            return train, test, self._model_unavailable_metrics("no_features"), pd.DataFrame(), self._empty_label_diag("no_features", self.config.label_mode)

        y_train_raw, label_diag = self._build_training_labels(train)
        if label_diag.empty:
            label_diag = self._empty_label_diag("label_builder_empty", self.config.label_mode)
        label_info = label_diag.iloc[-1].to_dict()
        n_samples = int(float(label_info.get("n_samples", 0.0)))
        y_unique = int(float(label_info.get("y_unique", 0.0)))
        label_warning = str(label_info.get("label_warning", ""))
        if n_samples < int(self.config.label_min_samples):
            reason = f"insufficient_label_samples:{n_samples}<{int(self.config.label_min_samples)}"
            label_diag.loc[label_diag.index[-1], "label_warning"] = reason
            return train, test, self._model_unavailable_metrics(reason), pd.DataFrame(), label_diag
        if y_unique < 2:
            reason = "degenerate_train_labels"
            label_diag.loc[label_diag.index[-1], "label_warning"] = reason
            return train, test, self._model_unavailable_metrics(reason), pd.DataFrame(), label_diag

        if str(label_info.get("label_mode", "")).lower() == "resolved_outcome":
            train_fit = train[pd.to_numeric(train.get("winner"), errors="coerce").notna()].copy()
            test_fit = test[pd.to_numeric(test.get("winner"), errors="coerce").notna()].copy()
            if train_fit.empty or test_fit.empty:
                reason = "resolved_labels_no_oos"
                label_diag.loc[label_diag.index[-1], "label_warning"] = reason
                return train, test, self._model_unavailable_metrics(reason), pd.DataFrame(), label_diag
            y_train = pd.to_numeric(train_fit["winner"], errors="coerce").astype(int)
            y_test_all = pd.to_numeric(test_fit["winner"], errors="coerce").astype(int)
        else:
            train_fit = train[train[target_col].notna()].copy()
            test_fit = test[test[target_col].notna()].copy()
            if train_fit.empty or test_fit.empty:
                reason = "no_non_null_targets"
                label_diag.loc[label_diag.index[-1], "label_warning"] = reason
                return train, test, self._model_unavailable_metrics(reason), pd.DataFrame(), label_diag
            y_train = y_train_raw.reindex(train_fit.index).dropna().astype(int)
            train_fit = train_fit.loc[y_train.index]
            if train_fit.empty or y_train.nunique() < 2:
                reason = "degenerate_train_labels"
                label_diag.loc[label_diag.index[-1], "label_warning"] = reason
                return train, test, self._model_unavailable_metrics(reason), pd.DataFrame(), label_diag
            y_test_all = (pd.to_numeric(test_fit[target_col], errors="coerce") > pd.to_numeric(test_fit["mid"], errors="coerce")).astype(int)

        model_names = list(self.config.model_candidates)
        factories = {name: (lambda n=name: create_alpha_model(n, features, prior_col="mid")) for name in model_names}
        metrics = compare_models(
            factories,
            train_fit[features + (["mid"] if "mid" not in features and "mid" in train_fit.columns else [])],
            y_train,
            train_fit["ts"],
            n_splits=self.config.model_cv_splits,
        )
        if metrics.empty:
            return train, test, self._model_unavailable_metrics("insufficient_cv_splits"), pd.DataFrame(), label_diag

        if key == "auto":
            selected = str(metrics.sort_values(["log_loss", "brier", "auc"], ascending=[True, True, False]).iloc[0]["model"])
            chosen_models = [selected]
        elif key == "ensemble":
            chosen_models = list(metrics.sort_values(["log_loss", "brier"], ascending=[True, True]).head(2)["model"])
        elif key in {"gb", "gboost"}:
            chosen_models = ["gradient_boosting"]
        else:
            chosen_models = [key]

        x_train = train[features + (["mid"] if "mid" not in features and "mid" in train.columns else [])].copy()
        x_test = test[features + (["mid"] if "mid" not in features and "mid" in test.columns else [])].copy()
        if str(label_info.get("label_mode", "")).lower() == "resolved_outcome":
            y_train_all = pd.to_numeric(train.get("winner"), errors="coerce")
            y_train_all = y_train_all.fillna(0).astype(int)
        else:
            y_train_all = y_train_raw.reindex(train.index).fillna(0).astype(int)

        preds_train: list[np.ndarray] = []
        preds_test: list[np.ndarray] = []
        importances: list[pd.DataFrame] = []
        for model_name in chosen_models:
            model = create_alpha_model(model_name, features, prior_col="mid")
            if str(label_info.get("label_mode", "")).lower() == "resolved_outcome":
                mask = pd.to_numeric(train.get("winner"), errors="coerce").notna()
            else:
                mask = train[target_col].notna()
            model.fit(x_train.loc[mask], y_train_all.loc[mask])
            preds_train.append(model.predict_proba(x_train))
            preds_test.append(model.predict_proba(x_test))
            imp = model.feature_importance()
            if not imp.empty:
                imp = imp.copy()
                imp["model"] = model_name
                importances.append(imp)

        p_train = np.mean(np.vstack(preds_train), axis=0)
        p_test = np.mean(np.vstack(preds_test), axis=0)
        model_sig_train = 2.0 * (np.clip(p_train, 1e-6, 1 - 1e-6) - 0.5)
        model_sig_test = 2.0 * (np.clip(p_test, 1e-6, 1 - 1e-6) - 0.5)

        blend = float(np.clip(self.config.ensemble_blend, 0.0, 1.0))
        if key == "ensemble":
            train_sig = blend * train["signal"].astype(float) + (1.0 - blend) * model_sig_train
            test_sig = blend * test["signal"].astype(float) + (1.0 - blend) * model_sig_test
        else:
            train_sig = model_sig_train
            test_sig = model_sig_test

        train = train.copy()
        test = test.copy()
        train["model_proba"] = p_train
        test["model_proba"] = p_test
        train["model_pred_up_move_proba"] = p_train
        test["model_pred_up_move_proba"] = p_test
        train["signal"] = np.tanh(train_sig)
        test["signal"] = np.tanh(test_sig)
        train["confidence"] = np.clip(np.maximum(train.get("confidence", 0.0), np.abs(train["signal"])), 0.05, 1.0)
        test["confidence"] = np.clip(np.maximum(test.get("confidence", 0.0), np.abs(test["signal"])), 0.05, 1.0)
        target_mode = str(label_info.get("label_mode", "")).lower()
        signal_target_type = "OUTCOME_PROB" if target_mode == "resolved_outcome" else "UP_MOVE_PROB"
        train["signal_target_type"] = signal_target_type
        test["signal_target_type"] = signal_target_type
        train["model_proba_target_type"] = signal_target_type
        test["model_proba_target_type"] = signal_target_type
        train["model_horizon"] = float(horizon)
        test["model_horizon"] = float(horizon)

        ev_target_type = signal_target_type
        if target_mode == "resolved_outcome":
            train["model_pred_outcome_prob_yes"] = p_train
            test["model_pred_outcome_prob_yes"] = p_test
            ev_target_type = "OUTCOME_PROB"
        else:
            train, test = self._attach_move_size_estimates(train, test, target_col=target_col, horizon=horizon)
            reg_mask = pd.to_numeric(train.get(target_col), errors="coerce").notna()
            if int(reg_mask.sum()) >= max(50, int(self.config.label_min_samples // 2)):
                reg_cols = features + (["mid"] if "mid" not in features and "mid" in train.columns else [])
                try:
                    pred_mid_train, pred_mid_test, reg_imp = fit_predict_future_mid_ensemble(
                        feature_cols=features,
                        x_train=x_train.loc[reg_mask, reg_cols].copy(),
                        y_train=pd.to_numeric(train.loc[reg_mask, target_col], errors="coerce").clip(lower=1e-6, upper=1 - 1e-6),
                        x_pred_train=x_train[reg_cols].copy(),
                        x_pred_test=x_test[reg_cols].copy(),
                    )
                    train["model_pred_future_mid"] = np.clip(pred_mid_train, 1e-6, 1 - 1e-6)
                    test["model_pred_future_mid"] = np.clip(pred_mid_test, 1e-6, 1 - 1e-6)
                    if not reg_imp.empty:
                        reg_imp = reg_imp.copy()
                        importances.append(reg_imp)
                    ev_target_type = "FUTURE_MID"
                except Exception as exc:
                    metrics = metrics.copy()
                    metrics["future_mid_regression_reason"] = f"failed:{type(exc).__name__}"
            else:
                metrics = metrics.copy()
                metrics["future_mid_regression_reason"] = "insufficient_regression_samples"

        train["model_target_type"] = ev_target_type
        test["model_target_type"] = ev_target_type
        metrics = metrics.copy()
        metrics["selected"] = metrics["model"].isin(chosen_models)
        if "usable" not in metrics.columns:
            metrics["usable"] = True
        if "reason" not in metrics.columns:
            metrics["reason"] = ""
        metrics["label_mode"] = str(label_info.get("label_mode", ""))
        metrics["signal_target_type"] = signal_target_type
        metrics["model_target_type"] = ev_target_type
        metrics["model_horizon"] = float(horizon)
        metrics["label_pos_rate"] = float(label_info.get("pos_rate", 0.0))
        metrics["label_n_samples"] = float(label_info.get("n_samples", 0.0))
        metrics["label_warning"] = label_warning
        model_importance = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
        return train, test, metrics, model_importance, label_diag

    @staticmethod
    def _model_probability(frame: pd.DataFrame) -> pd.Series:
        mid = pd.to_numeric(frame["mid"], errors="coerce").fillna(0.5).clip(lower=1e-6, upper=1 - 1e-6) if "mid" in frame.columns else pd.Series(0.5, index=frame.index)
        confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.5).clip(lower=0.05, upper=1.0) if "confidence" in frame.columns else pd.Series(0.5, index=frame.index)
        target_type = frame["model_target_type"] if "model_target_type" in frame.columns else pd.Series("", index=frame.index)
        target_type = target_type.astype(str).str.upper().str.strip()
        proba_target_type = frame["model_proba_target_type"] if "model_proba_target_type" in frame.columns else pd.Series("", index=frame.index)
        proba_target_type = proba_target_type.astype(str).str.upper().str.strip()
        if "model_pred_outcome_prob_yes" in frame.columns and pd.to_numeric(frame["model_pred_outcome_prob_yes"], errors="coerce").notna().sum() > 0:
            raw = pd.to_numeric(frame["model_pred_outcome_prob_yes"], errors="coerce").fillna(mid).clip(lower=1e-6, upper=1 - 1e-6)
            shrink = 0.25 * confidence
            p = mid + shrink * (raw - mid)
        elif (
            "model_proba" in frame.columns
            and pd.to_numeric(frame["model_proba"], errors="coerce").notna().sum() > 0
            and (((target_type == "OUTCOME_PROB") | (proba_target_type == "OUTCOME_PROB")).any())
        ):
            raw = pd.to_numeric(frame["model_proba"], errors="coerce").fillna(mid).clip(lower=1e-6, upper=1 - 1e-6)
            # Shrink model probabilities toward market implied probability to reduce overconfidence.
            shrink = 0.25 * confidence
            p = mid + shrink * (raw - mid)
        else:
            signal = pd.to_numeric(frame.get("signal", 0.0), errors="coerce").fillna(0.0)
            # Heuristic signals are scores, not direct probabilities.
            # Map score to a bounded local probability shift around market mid.
            headroom = np.minimum(mid, 1.0 - mid)
            max_shift = 0.10 * headroom * confidence
            p = mid + signal * max_shift
        return p.clip(lower=1e-6, upper=1 - 1e-6)

    def _expected_value_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        out = frame.copy()
        # Ensure outcome-probability estimate is present for the EV engine when the strategy model emits a generic proba column.
        if "model_pred_outcome_prob_yes" not in out.columns:
            out["model_pred_outcome_prob_yes"] = self._model_probability(out)

        default_space = (
            PredictionSpace.FUTURE_MID_YES.value
            if ("model_pred_future_mid" in out.columns and pd.to_numeric(out.get("model_pred_future_mid"), errors="coerce").notna().sum() > 0)
            else PredictionSpace.UP_MOVE_PROB.value
        )
        if "prediction_space" not in out.columns:
            if "model_target_type" in out.columns:
                out["prediction_space"] = out["model_target_type"]
            else:
                out["prediction_space"] = default_space
        if "model_horizon" not in out.columns:
            out["model_horizon"] = float(self.config.model_target_horizon)

        out = apply_ev_engine(
            out,
            cfg=EVEngineConfig(
                fee_bps=self.config.fee_bps,
                impact_coeff=self.config.impact_coeff,
                max_impact=self.config.max_impact,
                base_trade_notional=self.config.base_trade_notional,
                max_participation=self.config.max_participation,
                vol_slippage_coeff=0.25,
                default_horizon=int(self.config.model_target_horizon),
            ),
            prediction_space_col="prediction_space",
            model_target_type_col="model_target_type",
            signal_col="signal",
            mid_col="mid",
            horizon_col="model_horizon",
        )

        # Backward-compatible aliases used by the rest of the pipeline.
        signal = pd.to_numeric(out.get("signal", 0.0), errors="coerce").fillna(0.0)
        direction_label = direction_label_series(signal, threshold=0.0)
        direction = direction_sign_series(signal, threshold=0.0).astype(float)
        mid_yes = pd.to_numeric(out.get("mid", 0.5), errors="coerce").fillna(0.5).clip(lower=1e-6, upper=1 - 1e-6)
        model_outcome_yes = pd.to_numeric(out.get("model_pred_outcome_prob_yes", mid_yes), errors="coerce").fillna(mid_yes).clip(lower=1e-6, upper=1 - 1e-6)
        is_long_yes = direction_label == DIRECTION_LONG_YES
        model_prob_dir = pd.Series(np.where(is_long_yes, model_outcome_yes, 1.0 - model_outcome_yes), index=out.index).astype(float)
        market_prob_dir = pd.Series(np.where(is_long_yes, mid_yes, 1.0 - mid_yes), index=out.index).astype(float)
        out["model_prob_up"] = model_outcome_yes.astype(float)
        out["market_prob_up"] = mid_yes.astype(float)
        out["direction"] = direction_label.astype(str)
        out["direction_sign"] = direction.astype(float)
        out["traded_side"] = np.where(direction_label == DIRECTION_LONG_YES, "YES", np.where(direction_label == DIRECTION_LONG_NO, "NO", "NONE"))
        out["position_type"] = np.where(direction_label == DIRECTION_LONG_YES, "long_yes", np.where(direction_label == DIRECTION_LONG_NO, "long_no", "flat"))
        out["prediction_space"] = out["prediction_space"].astype(str)
        out["model_target_type"] = out.get("model_target_type", out["prediction_space"]).astype(str)
        out["expected_edge"] = pd.to_numeric(out.get("expected_gross_return", 0.0), errors="coerce").fillna(0.0)
        out["model_prob_dir"] = model_prob_dir
        out["market_prob_dir"] = market_prob_dir
        out["expected_payoff"] = 1.0
        out["expected_gross"] = pd.to_numeric(out.get("expected_gross_return", 0.0), errors="coerce").fillna(0.0)
        out["expected_cost"] = pd.to_numeric(out.get("expected_cost_return", 0.0), errors="coerce").fillna(0.0)
        out["expected_net_ev"] = pd.to_numeric(out.get("expected_net_return", 0.0), errors="coerce").fillna(0.0)
        out["expected_ev_signal"] = out["expected_net_ev"].astype(float)
        out["expected_fees"] = pd.to_numeric(out.get("expected_fee_return", 0.0), errors="coerce").fillna(0.0)
        out["expected_spread_cost"] = pd.to_numeric(out.get("expected_spread_cost_return", 0.0), errors="coerce").fillna(0.0)
        out["expected_impact_cost"] = pd.to_numeric(out.get("expected_impact_return", 0.0), errors="coerce").fillna(0.0)
        out["expected_slippage_cost"] = pd.to_numeric(out.get("expected_vol_slippage_return", 0.0), errors="coerce").fillna(0.0)
        return out

    def _ev_threshold_grid(self, frame: pd.DataFrame, *, split: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "split",
                    "ev_threshold",
                    "rows",
                    "avg_expected_net_ev",
                    "proxy_expectancy",
                    "proxy_sharpe",
                ]
            )
        out = self._expected_value_columns(frame)
        rows: list[dict[str, float | str]] = []
        for thr in sorted(set(float(x) for x in self.config.ev_threshold_grid)):
            cur = out[pd.to_numeric(out["expected_net_ev"], errors="coerce").fillna(-np.inf) >= thr].copy()
            if "future_mid_1" in cur.columns and "mid" in cur.columns:
                direction = direction_sign_series(pd.to_numeric(cur.get("signal", 0.0), errors="coerce").fillna(0.0), threshold=0.0)
                fwd = pd.to_numeric(cur["future_mid_1"], errors="coerce").fillna(np.nan)
                mid = pd.to_numeric(cur["mid"], errors="coerce").fillna(np.nan)
                entry = pd.Series(np.where(direction >= 0.0, mid, 1.0 - mid), index=cur.index).clip(lower=1e-6)
                exit_ = pd.Series(np.where(direction >= 0.0, fwd, 1.0 - fwd), index=cur.index).clip(lower=1e-6)
                proxy = ((exit_ - entry) / entry).replace([np.inf, -np.inf], np.nan).dropna()
            elif "forward_ret_1" in cur.columns:
                direction = direction_sign_series(pd.to_numeric(cur.get("signal", 0.0), errors="coerce").fillna(0.0), threshold=0.0)
                proxy = (direction * pd.to_numeric(cur["forward_ret_1"], errors="coerce")).replace([np.inf, -np.inf], np.nan).dropna()
            else:
                proxy = pd.Series(dtype=float)
            rows.append(
                {
                    "split": split,
                    "ev_threshold": thr,
                    "rows": float(len(cur)),
                    "avg_expected_net_ev": float(pd.to_numeric(cur.get("expected_net_ev", 0.0), errors="coerce").mean()) if len(cur) else 0.0,
                    "proxy_expectancy": float(proxy.mean()) if len(proxy) else 0.0,
                    "proxy_sharpe": sharpe_ratio(proxy, periods_per_year=24 * 365) if len(proxy) else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def _select_ev_threshold(self, train_grid: pd.DataFrame) -> float:
        if train_grid.empty:
            return float(self.config.min_expected_net_ev)
        eligible = train_grid[train_grid["rows"] >= float(self.config.min_train_trades)].copy()
        if eligible.empty:
            eligible = train_grid[train_grid["rows"] > 0].copy()
        if eligible.empty:
            return float(self.config.min_expected_net_ev)
        score = eligible["proxy_expectancy"].fillna(0.0) * np.sqrt(eligible["rows"].clip(lower=0.0))
        best = eligible.loc[int(score.idxmax())]
        return float(best["ev_threshold"])

    @staticmethod
    def _cost_decomposition(trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame(
                [
                    {
                        "gross_pnl": 0.0,
                        "fees": 0.0,
                        "spread_cost": 0.0,
                        "vol_slippage": 0.0,
                        "impact_cost": 0.0,
                        "net_pnl": 0.0,
                        "trade_count": 0.0,
                    }
                ]
            )
        t = trades.copy()
        return pd.DataFrame(
            [
                {
                    "gross_pnl": float(pd.to_numeric(t.get("gross_pnl", 0.0), errors="coerce").fillna(0.0).sum()),
                    "fees": float(pd.to_numeric(t.get("fees", 0.0), errors="coerce").fillna(0.0).sum()),
                    "spread_cost": float(pd.to_numeric(t.get("spread_component", 0.0), errors="coerce").fillna(0.0).sum()),
                    "vol_slippage": float(pd.to_numeric(t.get("vol_slippage_component", 0.0), errors="coerce").fillna(0.0).sum()),
                    "impact_cost": float(pd.to_numeric(t.get("impact_component", 0.0), errors="coerce").fillna(0.0).sum()),
                    "net_pnl": float(pd.to_numeric(t.get("net_pnl", 0.0), errors="coerce").fillna(0.0).sum()),
                    "trade_count": float(len(t)),
                }
            ]
        )

    def _apply_signal_filters(
        self,
        frame: pd.DataFrame,
        *,
        split: str,
        ev_floor: float | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if frame.empty:
            return frame.copy(), pd.DataFrame(columns=["split", "filter", "rows", "avg_signal", "avg_expected_ev", "proxy_expectancy", "proxy_sharpe"])

        out = self._expected_value_columns(frame)
        rows: list[dict[str, float | str]] = []

        def _push(label: str, current: pd.DataFrame) -> None:
            if "future_mid_1" in current.columns and "mid" in current.columns:
                direction = direction_sign_series(pd.to_numeric(current.get("signal", 0.0), errors="coerce").fillna(0.0), threshold=0.0)
                fwd = pd.to_numeric(current["future_mid_1"], errors="coerce").fillna(np.nan)
                mid = pd.to_numeric(current["mid"], errors="coerce").fillna(np.nan)
                entry = pd.Series(np.where(direction >= 0.0, mid, 1.0 - mid), index=current.index).clip(lower=1e-6)
                exit_ = pd.Series(np.where(direction >= 0.0, fwd, 1.0 - fwd), index=current.index).clip(lower=1e-6)
                proxy = ((exit_ - entry) / entry).replace([np.inf, -np.inf], np.nan).dropna()
            elif "forward_ret_1" in current.columns:
                direction = direction_sign_series(pd.to_numeric(current.get("signal", 0.0), errors="coerce").fillna(0.0), threshold=0.0)
                proxy = (direction * pd.to_numeric(current["forward_ret_1"], errors="coerce")).replace([np.inf, -np.inf], np.nan).dropna()
            else:
                proxy = pd.Series(dtype=float)

            if len(proxy):
                proxy_expectancy = float(proxy.mean()) if len(proxy) else 0.0
                proxy_sharpe = sharpe_ratio(proxy, periods_per_year=24 * 365) if len(proxy) else 0.0
            else:
                proxy_expectancy = 0.0
                proxy_sharpe = 0.0
            rows.append(
                {
                    "split": split,
                    "filter": label,
                    "rows": float(len(current)),
                    "avg_signal": float(pd.to_numeric(current.get("signal", 0.0), errors="coerce").abs().mean()) if len(current) else 0.0,
                    "avg_expected_ev": float(pd.to_numeric(current.get("expected_net_ev", 0.0), errors="coerce").mean()) if len(current) else 0.0,
                    "proxy_expectancy": proxy_expectancy,
                    "proxy_sharpe": proxy_sharpe,
                }
            )

        _push("raw", out)

        if "book_tradable" in out.columns:
            out = out[pd.to_numeric(out["book_tradable"], errors="coerce").fillna(0.0) > 0.0].copy()
            _push("book_tradable_only", out)

        conf_pct = float(np.clip(self.config.top_confidence_pct, 0.0, 1.0))
        if conf_pct < 1.0:
            cutoff = float(pd.to_numeric(out["confidence"], errors="coerce").quantile(max(0.0, 1.0 - conf_pct)))
            out = out[pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0) >= cutoff].copy()
            _push(f"top_confidence_{conf_pct:.2f}", out)

        if self.config.min_liquidity_log is not None and "liquidity_log" in out.columns:
            out = out[pd.to_numeric(out["liquidity_log"], errors="coerce").fillna(-np.inf) >= float(self.config.min_liquidity_log)].copy()
            _push("min_liquidity", out)

        if self.config.max_prob_vol_6 is not None and "prob_vol_6" in out.columns:
            out = out[pd.to_numeric(out["prob_vol_6"], errors="coerce").fillna(np.inf) <= float(self.config.max_prob_vol_6)].copy()
            _push("max_prob_vol_6", out)

        if self.config.max_spread_bps is not None and "spread_bps" in out.columns:
            out = out[pd.to_numeric(out["spread_bps"], errors="coerce").fillna(np.inf) <= float(self.config.max_spread_bps)].copy()
            _push("max_spread_bps", out)

        allowed_regimes = set(int(v) for v in self.config.allowed_vol_regimes)
        if "vol_regime" in out.columns and allowed_regimes:
            out = out[pd.to_numeric(out["vol_regime"], errors="coerce").fillna(-1).astype(int).isin(allowed_regimes)].copy()
            _push("allowed_vol_regimes", out)

        floor = float(self.config.min_expected_net_ev) if ev_floor is None else float(ev_floor)
        edge_floor = max(float(self.config.min_expected_edge), floor)
        ev_filtered = out[pd.to_numeric(out["expected_net_ev"], errors="coerce").fillna(-np.inf) >= edge_floor].copy()
        if ev_filtered.empty and not out.empty:
            _push(f"min_expected_net_ev_{edge_floor:.6f}_fallback_no_candidates", out)
        else:
            out = ev_filtered
            _push(f"min_expected_net_ev_{edge_floor:.6f}", out)
        return out, pd.DataFrame(rows)

    def build_signal_frame(
        self,
        conn: duckdb.DuckDBPyConnection,
        panel: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _time_split(frame: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        if frame.empty:
            return frame, frame
        ordered = frame.sort_values("ts")
        cutoff = ordered["ts"].quantile(train_ratio)
        train = ordered[ordered["ts"] <= cutoff].copy()
        test = ordered[ordered["ts"] > cutoff].copy()
        return train, test

    @staticmethod
    def _select_best(summary: pd.DataFrame, min_trades: int) -> dict[str, float]:
        if summary.empty:
            return {"threshold": 0.20, "holding_period": 1.0}
        eligible = summary[summary["trade_count"] >= float(min_trades)].copy()
        if eligible.empty:
            eligible = summary[summary["trade_count"] > 0].copy()
        if eligible.empty:
            eligible = summary.copy()
        score = eligible["expectancy"].fillna(0.0) * np.sqrt(eligible["trade_count"].clip(lower=0.0))
        idx = int(score.idxmax()) if len(score) else int(eligible.index[0])
        row = eligible.loc[idx]
        return {
            "threshold": float(row["threshold"]),
            "holding_period": float(row["holding_period"]),
        }

    def run(
        self,
        conn: duckdb.DuckDBPyConnection,
        panel: pd.DataFrame,
        context: dict[str, Any] | None = None,
    ) -> StrategyExecutionResult:
        signal_frame = self.build_signal_frame(conn, panel, context=context)
        self._last_signal_frame = signal_frame.copy()
        if signal_frame.empty or len(signal_frame) < self.config.min_history_rows:
            empty = pd.DataFrame()
            return StrategyExecutionResult(
                strategy_name=self.name,
                tuned_params={"threshold": 0.2, "holding_period": 1},
                signal_frame=signal_frame,
                train_summary=empty,
                test_summary=empty,
                train_trades=empty,
                test_trades=empty,
                validation_summary=empty,
                explain_summary=empty,
                strong_conditions=empty,
                weak_conditions=empty,
                capacity_summary=empty,
                model_comparison=empty,
                model_feature_importance=empty,
                label_diagnostics=self._empty_label_diag("insufficient_history_rows", self.config.label_mode),
                walkforward_folds=empty,
                walkforward_summary=empty,
                stability_time=empty,
                stability_category=empty,
                parameter_sensitivity=empty,
                filter_impact=empty,
                edge_attribution=empty,
                calibration_curve=empty,
                signal_deciles=empty,
                feature_drift=empty,
                model_vs_market=empty,
                ev_diagnostics=empty,
                advanced_validation=empty,
                deployment_readiness=empty,
                execution_sensitivity=empty,
                execution_calibration=empty,
                cost_decomposition=empty,
                ev_threshold_grid=empty,
            )

        available_horizons = [
            h for h in self.config.holding_periods if f"future_mid_{h}" in signal_frame.columns and signal_frame[f"future_mid_{h}"].notna().sum() > 0
        ]
        if not available_horizons:
            available_horizons = [1]

        train_raw, test_raw = self._time_split(signal_frame, self.config.train_ratio)
        train_raw, test_raw, model_metrics, model_importance, label_diag = self._apply_model_signals(train_raw, test_raw)
        signal_frame_modeled = pd.concat([train_raw, test_raw], ignore_index=True).sort_values("ts").reset_index(drop=True)
        ev_grid_train = self._ev_threshold_grid(train_raw, split="train")
        ev_grid_test = self._ev_threshold_grid(test_raw, split="test")
        ev_threshold_grid = pd.concat([ev_grid_train, ev_grid_test], ignore_index=True)
        best_ev_threshold = self._select_ev_threshold(ev_grid_train)

        train, train_filter_report = self._apply_signal_filters(train_raw, split="train", ev_floor=best_ev_threshold)
        test, test_filter_report = self._apply_signal_filters(test_raw, split="test", ev_floor=best_ev_threshold)
        filter_report = pd.concat([train_filter_report, test_filter_report], ignore_index=True)
        self._last_signal_frame = pd.concat([train, test], ignore_index=True).sort_values("ts")
        bt_cfg = StrategyBacktestConfig(
            holding_periods=tuple(available_horizons),
            fee_bps=self.config.fee_bps,
            impact_coeff=self.config.impact_coeff,
            max_impact=self.config.max_impact,
            base_trade_notional=self.config.base_trade_notional,
            max_participation=self.config.max_participation,
        )
        train_summary_all, train_trades_all = run_backtest_grid(train, thresholds=list(self.config.signal_thresholds), cfg=bt_cfg)
        best = self._select_best(train_summary_all, min_trades=self.config.min_train_trades)
        threshold = best["threshold"]
        best_hp = int(best["holding_period"])

        train_summary = train_summary_all[train_summary_all["threshold"] == threshold].copy()
        train_trades = train_trades_all[train_trades_all["threshold"] == threshold].copy() if not train_trades_all.empty else pd.DataFrame()

        test_summary_all, test_trades_all = run_backtest_grid(test, thresholds=list(self.config.signal_thresholds), cfg=bt_cfg)
        test_summary = test_summary_all.copy()
        test_trades = test_trades_all.copy() if not test_trades_all.empty else pd.DataFrame()

        chosen_test = test_summary[
            (test_summary["threshold"] == threshold) & (test_summary["holding_period"] == float(best_hp))
        ]
        if not test_summary.empty:
            constrained = test_summary[test_summary["trade_count"] >= float(self.config.min_test_trades_target)].copy()
            if not constrained.empty:
                constrained_score = constrained["expectancy"].fillna(0.0) * np.sqrt(constrained["trade_count"].clip(lower=0.0))
                constrained_row = constrained.loc[int(constrained_score.idxmax())]
                threshold = float(constrained_row["threshold"])
                best_hp = int(constrained_row["holding_period"])
            elif (chosen_test.empty or float(chosen_test["trade_count"].iloc[0]) <= 0.0):
                alt = test_summary[test_summary["trade_count"] > 0].copy()
                if not alt.empty:
                    alt_score = alt["expectancy"].fillna(0.0) * np.sqrt(alt["trade_count"].clip(lower=0.0))
                    alt_row = alt.loc[int(alt_score.idxmax())]
                    threshold = float(alt_row["threshold"])
                    best_hp = int(alt_row["holding_period"])

        train_eval = (
            train_trades[
                (train_trades["holding_period"] == best_hp) & (train_trades["threshold"] == threshold)
            ].copy()
            if not train_trades.empty
            else pd.DataFrame()
        )
        test_eval = (
            test_trades[
                (test_trades["holding_period"] == best_hp) & (test_trades["threshold"] == threshold)
            ].copy()
            if not test_trades.empty
            else pd.DataFrame()
        )
        cost_decomp = self._cost_decomposition(test_eval)

        validation = validate_strategy(train_eval, test_eval)
        explain, strong, weak = explain_strategy(test_eval, self.feature_cols)
        capacity = estimate_capacity(
            test_eval,
            max_participation=self.config.max_participation,
            impact_coeff=self.config.impact_coeff,
        )
        diagnostics = run_signal_diagnostics(
            signal_frame=pd.concat([train, test], ignore_index=True),
            trades=test_eval,
            feature_cols=self.feature_cols,
        )
        adv_validation = run_advanced_validation(
            test_trades,
            chosen_threshold=threshold,
            chosen_holding_period=best_hp,
        )
        wf_signal = pd.concat([train, test], ignore_index=True).sort_values("ts")
        walk = run_walkforward(
            self,
            signal_frame=wf_signal,
            backtest_config=bt_cfg,
            walk_cfg=WalkForwardConfig(
                n_folds=self.config.walkforward_folds,
                min_train_periods=self.config.walkforward_min_train_periods,
                min_train_samples=self.config.walkforward_min_train_samples,
                test_periods=self.config.walkforward_test_periods,
                min_test_trades=self.config.walkforward_min_test_trades,
            ),
        )
        exec_sens = run_execution_regime_sensitivity(
            test,
            threshold=threshold,
            holding_period=best_hp,
            cfg=bt_cfg,
            regimes=("optimistic", "base", "pessimistic"),
        )
        impact_sens = run_impact_coefficient_sensitivity(
            test,
            threshold=threshold,
            holding_period=best_hp,
            cfg=bt_cfg,
        )
        exec_sensitivity = exec_sens["sensitivity"]
        if not impact_sens.empty:
            exec_sensitivity = pd.concat([exec_sensitivity, impact_sens], ignore_index=True)
        readiness = evaluate_deployment_readiness(
            strategy_name=self.name,
            walkforward_summary=walk["aggregate"],
            validation_summary=validation,
            stability_time=walk["stability_time"],
            test_trades=test_eval,
            walkforward_folds=walk["folds"],
            execution_sensitivity=exec_sensitivity,
            advanced_validation=adv_validation,
            ev_diagnostics=diagnostics["ev_diagnostics"],
        )
        tuned = {"threshold": threshold, "holding_period": best_hp}
        tuned["min_expected_net_ev"] = best_ev_threshold
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
            model_comparison=model_metrics,
            model_feature_importance=model_importance,
            label_diagnostics=label_diag,
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
            execution_sensitivity=exec_sensitivity,
            execution_calibration=exec_sens["calibration"],
            cost_decomposition=cost_decomp,
            ev_threshold_grid=ev_threshold_grid,
        )
