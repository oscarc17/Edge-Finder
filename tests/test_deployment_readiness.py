from __future__ import annotations

import unittest

import pandas as pd

from polymarket_edge.research.deployment import evaluate_deployment_readiness


class DeploymentReadinessTests(unittest.TestCase):
    def test_summary_schema_and_pass_fail(self) -> None:
        walk_summary = pd.DataFrame([{"mean_expectancy": 0.001, "folds": 4.0, "valid_folds": 3.0}])
        walk_folds = pd.DataFrame([{"trade_count": 120.0, "valid_fold": True}, {"trade_count": 100.0, "valid_fold": True}])
        valid = pd.DataFrame(
            [
                {
                    "test_mean_return": 0.001,
                    "mean_return_ci_low": 0.0001,
                    "mean_return_ci_high": 0.002,
                    "ttest_pvalue": 0.02,
                    "permutation_pvalue": 0.03,
                }
            ]
        )
        stab = pd.DataFrame([{"avg_return": 0.001}, {"avg_return": 0.0005}, {"avg_return": 0.0002}])
        test_trades = pd.DataFrame({"trade_return": [0.001] * 210, "gross_return": [0.002] * 210})
        exec_sens = pd.DataFrame([{"execution_regime": "pessimistic", "avg_expected_ev": 0.0001}])
        adv = pd.DataFrame([{"white_pvalue": 0.1, "pbo": 0.2}])
        ev_diag = pd.DataFrame(
            [
                {
                    "ev_bin": 0.0,
                    "ev_monotonic_pass": 1.0,
                    "ev_spearman": 0.4,
                    "ev_model_valid": 1.0,
                }
            ]
        )

        out = evaluate_deployment_readiness(
            strategy_name="s",
            walkforward_summary=walk_summary,
            walkforward_folds=walk_folds,
            validation_summary=valid,
            stability_time=stab,
            test_trades=test_trades,
            execution_sensitivity=exec_sens,
            advanced_validation=adv,
            ev_diagnostics=ev_diag,
        )
        summary = out[out["criterion"] == "summary"]
        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertTrue(bool(row["passed"]))
        self.assertEqual(str(row["decision"]), "PASS")
        required = [
            "test_mean_return",
            "ci_low",
            "ci_high",
            "n_test_trades",
            "n_walkforward_trades",
            "walkforward_valid_folds",
            "walkforward_mean_expectancy",
            "stability_time_positive_ratio",
            "stability_score",
            "pessimistic_avg_expected_ev",
            "advanced_white_pvalue",
            "advanced_pbo",
            "ev_monotonic_pass",
            "ev_monotonic_spearman",
            "ev_spearman",
            "hit_rate_pos_ev",
            "ev_model_valid",
        ]
        for col in required:
            self.assertIn(col, out.columns)
            self.assertFalse(pd.isna(row[col]))

    def test_pbo_fail(self) -> None:
        walk_summary = pd.DataFrame([{"mean_expectancy": 0.001, "folds": 4.0, "valid_folds": 3.0}])
        walk_folds = pd.DataFrame([{"trade_count": 200.0, "valid_fold": True}])
        valid = pd.DataFrame(
            [
                {
                    "test_mean_return": 0.001,
                    "mean_return_ci_low": 0.0001,
                    "mean_return_ci_high": 0.002,
                    "ttest_pvalue": 0.02,
                    "permutation_pvalue": 0.03,
                }
            ]
        )
        stab = pd.DataFrame([{"avg_return": 0.001}, {"avg_return": 0.0005}])
        test_trades = pd.DataFrame({"trade_return": [0.001] * 220, "gross_return": [0.002] * 220})
        exec_sens = pd.DataFrame([{"execution_regime": "pessimistic", "avg_expected_ev": 0.0001}])
        adv = pd.DataFrame([{"white_pvalue": 0.1, "pbo": 0.8}])
        ev_diag = pd.DataFrame(
            [
                {
                    "ev_bin": 0.0,
                    "ev_monotonic_pass": 1.0,
                    "ev_spearman": 0.3,
                    "ev_model_valid": 1.0,
                }
            ]
        )

        out = evaluate_deployment_readiness(
            strategy_name="s",
            walkforward_summary=walk_summary,
            walkforward_folds=walk_folds,
            validation_summary=valid,
            stability_time=stab,
            test_trades=test_trades,
            execution_sensitivity=exec_sens,
            advanced_validation=adv,
            ev_diagnostics=ev_diag,
        )
        dec = out[out["criterion"] == "deployment_decision"].iloc[0]
        self.assertFalse(bool(dec["passed"]))
        self.assertEqual(str(dec["decision"]), "FAIL")
        summary = out[out["criterion"] == "summary"].iloc[0]
        self.assertIn("pbo", str(summary["notes"]).lower())


if __name__ == "__main__":
    unittest.main()
