"""单元测试：optimization_trace_view.compute_live_stats（不依赖 RELAP）。"""

from __future__ import annotations

import math
import unittest

from optimization_trace_view import compute_live_stats


class TestComputeLiveStats(unittest.TestCase):
    def test_empty_history(self) -> None:
        s = compute_live_stats([])
        self.assertEqual(s["total"], 0)
        self.assertEqual(s["failed"], 0)
        self.assertIsNone(s["latest_obj"])
        self.assertIsNone(s["best_obj"])
        self.assertIsNone(s["delta_vs_prev_best"])

    def test_first_evaluation_no_prev_best(self) -> None:
        h = [{"evaluation": 1, "parameter_value": 10.0, "objective_value": 5.0}]
        s = compute_live_stats(h)
        self.assertEqual(s["total"], 1)
        self.assertAlmostEqual(s["latest_obj"], 5.0)
        self.assertAlmostEqual(s["best_obj"], 5.0)
        self.assertAlmostEqual(s["best_param"], 10.0)
        self.assertEqual(s["best_eval_id"], 1)
        self.assertIsNone(s["delta_vs_prev_best"])

    def test_improvement_detected(self) -> None:
        h = [
            {"evaluation": 1, "parameter_value": 10.0, "objective_value": 5.0},
            {"evaluation": 2, "parameter_value": 20.0, "objective_value": 3.0},
        ]
        s = compute_live_stats(h)
        self.assertAlmostEqual(s["best_obj"], 3.0)
        self.assertAlmostEqual(s["best_param"], 20.0)
        self.assertEqual(s["best_eval_id"], 2)
        self.assertAlmostEqual(s["delta_vs_prev_best"], -2.0)  # 改进 2

    def test_regression_detected(self) -> None:
        h = [
            {"evaluation": 1, "parameter_value": 10.0, "objective_value": 3.0},
            {"evaluation": 2, "parameter_value": 50.0, "objective_value": 7.5},
        ]
        s = compute_live_stats(h)
        self.assertAlmostEqual(s["best_obj"], 3.0)
        self.assertAlmostEqual(s["best_param"], 10.0)
        self.assertEqual(s["best_eval_id"], 1)
        self.assertAlmostEqual(s["delta_vs_prev_best"], 4.5)  # 比之前最优差 4.5

    def test_failed_evaluations_counted_and_skipped(self) -> None:
        h = [
            {"evaluation": 1, "parameter_value": 10.0, "objective_value": 5.0},
            {"evaluation": 2, "parameter_value": 999.0,
             "objective_value": float("inf"), "relap_failed": True},
            {"evaluation": 3, "parameter_value": 30.0, "objective_value": 4.0},
        ]
        s = compute_live_stats(h)
        self.assertEqual(s["failed"], 1)
        self.assertAlmostEqual(s["best_obj"], 4.0)
        self.assertEqual(s["best_eval_id"], 3)
        # 最近一次 obj=4.0；prev_best (excl latest) = 5.0；delta = -1.0（改进）
        self.assertAlmostEqual(s["delta_vs_prev_best"], -1.0)

    def test_post_run_error_counted_as_failed(self) -> None:
        h = [
            {"evaluation": 1, "parameter_value": 10.0, "objective_value": 5.0},
            {"evaluation": 2, "parameter_value": 20.0,
             "objective_value": float("inf"),
             "post_run_error": "extract failed"},
        ]
        s = compute_live_stats(h)
        self.assertEqual(s["failed"], 1)
        self.assertAlmostEqual(s["best_obj"], 5.0)
        self.assertIsNone(s["latest_obj"])  # 最近一次 inf → None

    def test_target_band_records_use_abs_error(self) -> None:
        h = [
            {"iteration": 1, "parameter_value": 10.0,
             "criterion_value": 1.0, "error": 0.4, "success": False},
            {"iteration": 2, "parameter_value": 12.0,
             "criterion_value": 0.95, "error": -0.05, "success": True},
        ]
        s = compute_live_stats(h)
        # objective_of() 用 |error|: 0.4 vs 0.05 → best = 0.05 at iter 2
        self.assertAlmostEqual(s["best_obj"], 0.05)
        self.assertEqual(s["best_eval_id"], 2)
        # delta = 0.05 - 0.4 = -0.35（改进）
        self.assertAlmostEqual(s["delta_vs_prev_best"], -0.35)


if __name__ == "__main__":
    unittest.main()
