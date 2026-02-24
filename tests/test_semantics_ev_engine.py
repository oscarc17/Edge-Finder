from __future__ import annotations

import unittest

import pandas as pd

from polymarket_edge.research.ev_engine import EVEngineConfig, apply_ev_engine
from polymarket_edge.research.semantics import (
    PredictionSpace,
    expected_return_comparable,
    validate_prediction_contract,
)


class SemanticsEVEngineTests(unittest.TestCase):
    def test_future_mid_expected_return_flips_with_direction(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "signal": 0.8,
                    "mid": 0.55,
                    "spread": 0.01,
                    "depth_total": 1000.0,
                    "prob_vol_6": 0.01,
                    "confidence": 0.9,
                    "model_target_type": "FUTURE_MID",
                    "model_horizon": 1,
                    "model_pred_future_mid": 0.60,
                },
                {
                    "signal": -0.8,
                    "mid": 0.55,
                    "spread": 0.01,
                    "depth_total": 1000.0,
                    "prob_vol_6": 0.01,
                    "confidence": 0.9,
                    "model_target_type": "FUTURE_MID",
                    "model_horizon": 1,
                    "model_pred_future_mid": 0.60,
                },
            ]
        )
        out = apply_ev_engine(frame, cfg=EVEngineConfig(default_horizon=1))
        self.assertGreater(float(out.iloc[0]["expected_gross_return"]), 0.0)
        self.assertLess(float(out.iloc[1]["expected_gross_return"]), 0.0)
        ok, issues = validate_prediction_contract(out, require_expected_return=True)
        self.assertTrue(ok, msg=str(issues))

    def test_outcome_prob_expected_return_sign(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "signal": 0.5,
                    "mid": 0.40,
                    "spread": 0.01,
                    "depth_total": 2000.0,
                    "prob_vol_6": 0.01,
                    "confidence": 0.8,
                    "prediction_space": PredictionSpace.OUTCOME_PROB_YES.value,
                    "model_target_type": "OUTCOME_PROB",
                    "model_pred_outcome_prob_yes": 0.60,
                },
                {
                    "signal": -0.5,
                    "mid": 0.40,
                    "spread": 0.01,
                    "depth_total": 2000.0,
                    "prob_vol_6": 0.01,
                    "confidence": 0.8,
                    "prediction_space": PredictionSpace.OUTCOME_PROB_YES.value,
                    "model_target_type": "OUTCOME_PROB",
                    "model_pred_outcome_prob_yes": 0.60,
                },
            ]
        )
        out = apply_ev_engine(frame, cfg=EVEngineConfig(default_horizon=1))
        self.assertGreater(float(out.iloc[0]["expected_gross_return"]), 0.0)
        self.assertLess(float(out.iloc[1]["expected_gross_return"]), 0.0)

    def test_expected_return_units_comparable_to_trade_return(self) -> None:
        frame = pd.DataFrame(
            {
                "signal": [0.8, -0.7, 0.4, -0.2],
                "mid": [0.52, 0.48, 0.60, 0.30],
                "spread": [0.01, 0.01, 0.02, 0.01],
                "depth_total": [1000, 1100, 800, 700],
                "prob_vol_6": [0.01, 0.02, 0.03, 0.01],
                "confidence": [0.9, 0.8, 0.7, 0.6],
                "model_target_type": ["UP_MOVE_PROB"] * 4,
                "model_pred_up_move_proba": [0.7, 0.3, 0.6, 0.4],
                "expected_move_size": [0.03, 0.03, 0.04, 0.02],
            }
        )
        out = apply_ev_engine(frame, cfg=EVEngineConfig(default_horizon=1))
        out["trade_return"] = out["expected_return"] * 0.5
        self.assertTrue(expected_return_comparable(out))


if __name__ == "__main__":
    unittest.main()

