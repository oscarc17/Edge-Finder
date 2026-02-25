from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from polymarket_edge.research.direction import (
    DIRECTION_FLAT,
    DIRECTION_LONG_NO,
    DIRECTION_LONG_YES,
    direction_from_signal,
)
from polymarket_edge.research.diagnostics import run_signal_diagnostics
from polymarket_edge.strategies.base import BaseStrategy


class EVDirectionalityTests(unittest.TestCase):
    def test_direction_from_signal_threshold(self) -> None:
        self.assertEqual(direction_from_signal(0.20, threshold=0.10), DIRECTION_LONG_YES)
        self.assertEqual(direction_from_signal(-0.20, threshold=0.10), DIRECTION_LONG_NO)
        self.assertEqual(direction_from_signal(0.05, threshold=0.10), DIRECTION_FLAT)

    def test_expected_ev_flips_when_direction_flips(self) -> None:
        strat = BaseStrategy()
        frame = pd.DataFrame(
            [
                {
                    "signal": 0.8,
                    "model_proba": 0.60,
                    "mid": 0.55,
                    "spread": 0.01,
                    "spread_bps": 180.0,
                    "depth_total": 2000.0,
                    "confidence": 0.8,
                },
                {
                    "signal": -0.8,
                    "model_proba": 0.60,
                    "mid": 0.55,
                    "spread": 0.01,
                    "spread_bps": 180.0,
                    "depth_total": 2000.0,
                    "confidence": 0.8,
                },
            ]
        )
        out = strat._expected_value_columns(frame)
        gross_long_yes = float(out.iloc[0]["expected_gross"])
        gross_long_no = float(out.iloc[1]["expected_gross"])
        self.assertGreater(gross_long_yes, 0.0)
        self.assertLess(gross_long_no, 0.0)

    def test_ev_deciles_show_positive_monotonic_signal(self) -> None:
        rng = np.random.default_rng(7)
        n = 200
        ev = np.linspace(-0.01, 0.01, n)
        realized = ev + rng.normal(0.0, 0.001, size=n)
        ts = pd.date_range("2026-01-01", periods=n, freq="h")
        trades = pd.DataFrame(
            {
                "ts": ts,
                "signal": np.sign(ev),
                "mid": 0.5,
                "trade_return": realized,
                "expected_net_return": ev,
            }
        )
        diag = run_signal_diagnostics(signal_frame=trades.copy(), trades=trades, feature_cols=[])
        ev_diag = diag["ev_diagnostics"]
        total = ev_diag.iloc[0]
        self.assertGreater(float(total["ev_spearman"]), 0.0)
        self.assertGreaterEqual(float(total["ev_monotonic_pass"]), 1.0)
        self.assertGreaterEqual(float(total["ev_model_valid"]), 1.0)
        self.assertEqual(str(total["ev_expected_col"]), "expected_net_return")


if __name__ == "__main__":
    unittest.main()
