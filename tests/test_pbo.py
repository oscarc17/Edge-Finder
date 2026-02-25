from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from polymarket_edge.research.advanced_validation import probability_backtest_overfitting_with_debug


class PBOTests(unittest.TestCase):
    def test_pbo_returns_debug_and_valid_range(self) -> None:
        rng = np.random.default_rng(7)
        ts = pd.date_range("2026-01-01", periods=120, freq="h")
        # candidate_a good/stable, b noisy, c poor
        mat = pd.DataFrame(
            {
                "a": 0.001 + rng.normal(0, 0.0005, size=len(ts)),
                "b": rng.normal(0, 0.0010, size=len(ts)),
                "c": -0.0005 + rng.normal(0, 0.0005, size=len(ts)),
            },
            index=ts,
        )
        summary, debug = probability_backtest_overfitting_with_debug(mat, n_slices=8, max_trials=50, seed=11)
        self.assertIn("pbo", summary)
        self.assertIn("pbo_computable", summary)
        self.assertTrue(np.isfinite(float(summary["pbo"])))
        self.assertGreaterEqual(float(summary["pbo"]), 0.0)
        self.assertLessEqual(float(summary["pbo"]), 1.0)
        self.assertEqual(float(summary["pbo_computable"]), 1.0)
        self.assertFalse(debug.empty)
        self.assertIn("lambda", debug.columns)
        self.assertIn("chosen_oos_rank", debug.columns)
        self.assertTrue((debug["status"].astype(str) == "ok").any())

    def test_pbo_insufficient_returns_nan_not_one(self) -> None:
        mat = pd.DataFrame({"a": [0.1, 0.2], "b": [0.0, -0.1]})
        summary, debug = probability_backtest_overfitting_with_debug(mat, n_slices=8, max_trials=50, seed=11)
        self.assertTrue(np.isnan(float(summary["pbo"])))
        self.assertEqual(float(summary["pbo_computable"]), 0.0)
        self.assertFalse(debug.empty)


if __name__ == "__main__":
    unittest.main()

