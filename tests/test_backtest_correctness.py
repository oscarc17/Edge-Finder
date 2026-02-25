from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from polymarket_edge.research.backtest import StrategyBacktestConfig, run_backtest_grid
from polymarket_edge.research.direction import DIRECTION_LONG_NO, DIRECTION_LONG_YES, direction_from_signal
from polymarket_edge.research.execution import ExecutionCostInputs, compute_trade_return


def _toy_signal_frame(
    *,
    n: int = 1000,
    seed: int = 7,
    predictor: str = "perfect",  # perfect | random
    move_size: float = 0.03,
    spread: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mid = rng.uniform(0.20, 0.80, size=n)
    realized_move_sign = rng.choice([-1.0, 1.0], size=n)
    future_mid = np.clip(mid + realized_move_sign * float(move_size), 1e-4, 1 - 1e-4)
    if predictor == "perfect":
        signal = np.sign(future_mid - mid)
    elif predictor == "random":
        signal = rng.choice([-1.0, 1.0], size=n)
    else:
        raise ValueError("unknown predictor")

    ts = pd.date_range("2026-01-01", periods=n, freq="min")
    frame = pd.DataFrame(
        {
            "ts": ts,
            "market_id": np.arange(n) % 25,
            "token_id": np.arange(n).astype(str),
            "category": "toy",
            "signal": signal,
            "confidence": 1.0,
            "mid": mid,
            "spread": float(spread),
            "spread_bps": np.where(mid > 0, float(spread) / mid * 10_000.0, 0.0),
            "depth_total": 100.0,
            "trade_notional_h": 1_000_000.0,
            "prob_vol_6": 0.0,
            "future_mid_1": future_mid,
            "future_spread_1": float(spread),
            "book_tradable": 1.0,
            "model_target_type": "FUTURE_MID",
            "model_pred_future_mid": future_mid if predictor == "perfect" else mid,
            "model_horizon": 1.0,
        }
    )
    return frame


class BacktestCorrectnessTests(unittest.TestCase):
    def _run_toy(
        self,
        frame: pd.DataFrame,
        *,
        fee_bps: float = 0.0,
        spread_impact_vol_zero: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = StrategyBacktestConfig(
            holding_periods=(1,),
            fee_bps=float(fee_bps),
            impact_coeff=0.0 if spread_impact_vol_zero else 0.10,
            max_impact=0.0 if spread_impact_vol_zero else 0.05,
            base_trade_notional=20.0,
            max_participation=1.0,
            vol_slippage_coeff=0.0 if spread_impact_vol_zero else 0.25,
            fill_alpha_liquidity=25.0,
            fill_beta_spread=0.0,
            fill_gamma_volume=0.0,
            snapshot_hours=1.0,
            execution_regime="base",
        )
        summary, trades = run_backtest_grid(frame, thresholds=[0.0], cfg=cfg)
        return summary, trades

    def test_direction_sanity_predicted_up_profits_in_toy_world(self) -> None:
        self.assertEqual(direction_from_signal(0.3), DIRECTION_LONG_YES)
        calc = compute_trade_return(
            entry=0.40,
            exit=0.50,
            side=DIRECTION_LONG_YES,
            costs=ExecutionCostInputs(fee_rate=0.0),
            qty=10.0,
        )
        self.assertGreater(float(calc.iloc[0]["trade_return"]), 0.0)
        calc_flip = compute_trade_return(
            entry=0.40,
            exit=0.50,
            side=DIRECTION_LONG_NO,
            costs=ExecutionCostInputs(fee_rate=0.0),
            qty=10.0,
        )
        self.assertLess(float(calc_flip.iloc[0]["trade_return"]), 0.0)

    def test_perfect_predictor_zero_costs_is_profitable(self) -> None:
        frame = _toy_signal_frame(n=1500, predictor="perfect", move_size=0.03, spread=0.0)
        summary, trades = self._run_toy(frame, fee_bps=0.0, spread_impact_vol_zero=True)
        self.assertGreater(float(summary.iloc[0]["trade_count"]), 1000.0)
        self.assertGreater(float(summary.iloc[0]["expectancy_per_trade"]), 0.02)
        self.assertGreater(float(trades["trade_return"].mean()), 0.02)

    def test_random_predictor_zero_costs_is_near_zero(self) -> None:
        frame = _toy_signal_frame(n=4000, predictor="random", move_size=0.03, spread=0.0)
        summary, trades = self._run_toy(frame, fee_bps=0.0, spread_impact_vol_zero=True)
        self.assertGreater(float(summary.iloc[0]["trade_count"]), 2000.0)
        self.assertLess(abs(float(summary.iloc[0]["expectancy_per_trade"])), 0.005)
        self.assertLess(abs(float(trades["trade_return"].mean())), 0.005)

    def test_perfect_predictor_with_costs_stays_profitable_if_edge_exceeds_costs(self) -> None:
        frame = _toy_signal_frame(n=1500, predictor="perfect", move_size=0.05, spread=0.002)
        summary, trades = self._run_toy(frame, fee_bps=5.0, spread_impact_vol_zero=True)
        self.assertGreater(float(summary.iloc[0]["trade_count"]), 1000.0)
        self.assertGreater(float(summary.iloc[0]["expectancy_per_trade"]), 0.02)
        self.assertGreater(float(trades["trade_return"].mean()), 0.02)


if __name__ == "__main__":
    unittest.main()

