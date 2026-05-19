"""单元测试：auto_optimize 的多参数 / 多目标支持（不依赖 RELAP）。"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from auto_optimize import (
    _compute_recovery_candidate,
    format_value,
    normalize_objectives_config,
    normalize_parameters_config,
    run_optimization,
    update_parameter_in_indta,
    update_parameters_in_indta,
    run_optimization_minimize_time_integral,
    validate_integral_optimization_config,
)


# ---- normalize_parameters_config / normalize_objectives_config ---------------

class TestNormalizeParametersConfig(unittest.TestCase):
    def test_legacy_singular_wraps(self) -> None:
        cfg = {"parameter": {"name": "p1", "min_value": 0, "max_value": 1}}
        out = normalize_parameters_config(cfg)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "p1")

    def test_new_plural_passthrough(self) -> None:
        cfg = {"parameters": [{"name": "p1"}, {"name": "p2"}]}
        out = normalize_parameters_config(cfg)
        self.assertEqual([p["name"] for p in out], ["p1", "p2"])

    def test_missing_raises(self) -> None:
        with self.assertRaises(KeyError):
            normalize_parameters_config({})

    def test_empty_list_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_parameters_config({"parameters": []})


class TestNormalizeObjectivesConfig(unittest.TestCase):
    def test_legacy_integral_objective_wraps(self) -> None:
        cfg = {
            "integral_objective": {
                "value_column_index": 16,
                "reference_value": 1.5e7,
                "deviation": "absolute",
            }
        }
        out = normalize_objectives_config(cfg)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["value_column_index"], 16)
        self.assertAlmostEqual(out[0]["reference_value"], 1.5e7)
        self.assertEqual(out[0]["weight"], 1.0)
        self.assertEqual(out[0]["scale"], 1.0)

    def test_new_objectives_plural_passthrough(self) -> None:
        cfg = {
            "objectives": [
                {"value_column_index": 16, "reference_value": 1.5e7, "weight": 0.7},
                {"value_column_index": 14, "reference_value": 1e6, "weight": 0.3},
            ]
        }
        out = normalize_objectives_config(cfg)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0]["weight"], 0.7)


def _valid_integral_config():
    return {
        "env": "test",
        "objective": "minimize_time_integral",
        "parameters": [{
            "name": "p",
            "line_key": "1",
            "column_index": 1,
            "initial_value": 0.5,
            "min_value": 0.0,
            "max_value": 1.0,
            "step": 0.1,
        }],
        "objectives": [{
            "name": "o",
            "value_column_index": 2,
            "reference_value": 0.0,
            "deviation": "absolute",
            "weight": 1.0,
            "scale": 1.0,
        }],
        "integral_objective": {
            "time_column_index": 0,
            "csv_relative_path": "dummy.csv",
        },
        "integral_optimizer": {
            "optimizer": "genetic",
            "max_function_evaluations": 4,
            "max_relap_retries": 0,
            "random_seed": 1,
            "ga_crossover_probability": 0.9,
            "ga_mutation_scale": 0.1,
        },
    }


class TestValidateIntegralOptimizationConfig(unittest.TestCase):
    def test_accepts_valid_multi_objective_config(self) -> None:
        cfg = _valid_integral_config()
        cfg["parameters"].append({
            "name": "p2",
            "line_key": "2",
            "column_index": 2,
            "initial_value": 2.0,
            "min_value": 1.0,
            "max_value": 3.0,
            "step": 0.1,
        })
        cfg["objectives"].append({
            "name": "o2",
            "value_column_index": 3,
            "reference_value": 1.0,
            "deviation": "squared",
            "weight": 0.5,
            "scale": 10.0,
        })
        validate_integral_optimization_config(cfg)

    def test_rejects_empty_parameters(self) -> None:
        cfg = _valid_integral_config()
        cfg["parameters"] = []
        with self.assertRaises(ValueError):
            validate_integral_optimization_config(cfg)

    def test_rejects_invalid_bounds(self) -> None:
        cfg = _valid_integral_config()
        cfg["parameters"][0]["min_value"] = 1.0
        cfg["parameters"][0]["max_value"] = 1.0
        with self.assertRaisesRegex(ValueError, "min_value"):
            validate_integral_optimization_config(cfg)

    def test_rejects_initial_outside_bounds(self) -> None:
        cfg = _valid_integral_config()
        cfg["parameters"][0]["initial_value"] = 2.0
        with self.assertRaisesRegex(ValueError, "initial_value"):
            validate_integral_optimization_config(cfg)

    def test_rejects_negative_column_indices(self) -> None:
        cfg = _valid_integral_config()
        cfg["objectives"][0]["value_column_index"] = -1
        with self.assertRaisesRegex(ValueError, "value_column_index"):
            validate_integral_optimization_config(cfg)

    def test_rejects_zero_scale(self) -> None:
        cfg = _valid_integral_config()
        cfg["objectives"][0]["scale"] = 0.0
        with self.assertRaisesRegex(ValueError, "scale"):
            validate_integral_optimization_config(cfg)

    def test_rejects_non_positive_max_evaluations(self) -> None:
        cfg = _valid_integral_config()
        cfg["integral_optimizer"]["max_function_evaluations"] = 0
        with self.assertRaisesRegex(ValueError, "max_function_evaluations"):
            validate_integral_optimization_config(cfg)

    def test_run_optimization_rejects_retired_target_band(self) -> None:
        cfg = _valid_integral_config()
        cfg["objective"] = "target_band"
        with self.assertRaisesRegex(ValueError, "target_band"):
            run_optimization(config_override=cfg)


# ---- update_parameters_in_indta ---------------------------------------------

class TestUpdateParametersInIndta(unittest.TestCase):
    def test_writes_two_parameters_on_different_lines(self) -> None:
        lines = [
            "20510100  przp_err  sum  100.0  1.0  0\n",
            "20510200  another    sum  50.0   2.0  0\n",
        ]
        cfgs = [
            {"line_key": "20510100", "column_index": 3},
            {"line_key": "20510200", "column_index": 3},
        ]
        out = update_parameters_in_indta(lines, cfgs, [123.45, 67.89])
        self.assertIn("123.45", out[0])
        self.assertIn("67.89", out[1])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            update_parameters_in_indta(
                ["x"],
                [{"line_key": "1", "column_index": 0}],
                [1.0, 2.0],  # 2 个值但 1 个 cfg
            )

    def test_single_param_equivalent_to_legacy(self) -> None:
        lines = ["20510100  przp_err  sum  100.0  1.0  0\n"]
        cfg = {"line_key": "20510100", "column_index": 3}

        v_legacy = update_parameter_in_indta(lines, cfg, 7.5)
        v_multi = update_parameters_in_indta(lines, [cfg], [7.5])
        self.assertEqual(v_legacy, v_multi)

    def test_integer_float_is_written_with_decimal_point(self) -> None:
        lines = ["20510100  przp_err  sum  100.0  0.0  1\n"]
        cfg = {"line_key": "20510100", "column_index": 3}

        out = update_parameter_in_indta(lines, cfg, 10.0)

        self.assertIn("sum  10.0  0.0", out[0])
        self.assertEqual(format_value(1_000_000.0), "1.0e+06")


# ---- _compute_recovery_candidate（N-D 升级） --------------------------------

class TestRecoveryCandidateNd(unittest.TestCase):
    def test_2d_bisection_toward_last_safe(self) -> None:
        new = _compute_recovery_candidate(
            failed_x=np.array([200.0, 200.0]),
            last_safe_x=np.array([10.0, 10.0]),
            initial_value=np.array([50.0, 50.0]),
            bracket=[(0.0, 500.0), (0.0, 500.0)],
        )
        # last_safe 优先：new = (failed + last_safe) / 2 = (105, 105)
        self.assertIsNotNone(new)
        np.testing.assert_allclose(new, [105.0, 105.0])

    def test_2d_falls_back_to_initial(self) -> None:
        new = _compute_recovery_candidate(
            failed_x=np.array([200.0, 200.0]),
            last_safe_x=None,
            initial_value=np.array([10.0, 30.0]),
            bracket=[(0.0, 500.0), (0.0, 500.0)],
        )
        np.testing.assert_allclose(new, [105.0, 115.0])

    def test_2d_returns_none_when_anchors_coincide(self) -> None:
        new = _compute_recovery_candidate(
            failed_x=np.array([5.0, 5.0]),
            last_safe_x=None,
            initial_value=np.array([5.0, 5.0]),
            bracket=[(5.0, 5.0), (5.0, 5.0)],
        )
        self.assertIsNone(new)

    def test_1d_legacy_path_unchanged(self) -> None:
        # 1D 标量 API 必须保持现有行为（避免回归 test_recovery_candidate）
        new = _compute_recovery_candidate(
            failed_x=200.0, last_safe_x=10.0,
            initial_value=50.0, bracket=(0.0, 500.0),
        )
        self.assertAlmostEqual(new, 105.0)

    def test_2d_clamps_to_bounds(self) -> None:
        new = _compute_recovery_candidate(
            failed_x=np.array([1.0, 1.0]),
            last_safe_x=np.array([-1000.0, 1000.0]),  # 故意越界
            initial_value=np.array([5.0, 5.0]),
            bracket=[(0.0, 500.0), (0.0, 500.0)],
        )
        self.assertGreaterEqual(new[0], 0.0)
        self.assertLessEqual(new[0], 500.0)
        self.assertGreaterEqual(new[1], 0.0)
        self.assertLessEqual(new[1], 500.0)


class TestGeneticOptimizerIntegration(unittest.TestCase):
    def test_ga_returns_pareto_front_and_weighted_representative(self) -> None:
        current = {"x": np.array([0.0])}

        def fake_try_relap(x_request, **kwargs):
            x = np.atleast_1d(np.asarray(x_request, dtype=float)).copy()
            current["x"] = x
            return x, {"failed": False, "last_line": "ok", "matched_pattern": None}, [x]

        def fake_weighted_objective(csv_path, time_col, objectives):
            x = float(current["x"][0])
            parts = [x * x, (x - 2.0) ** 2]
            return sum(parts), parts

        cfg = {
            "env": "test",
            "objective": "minimize_time_integral",
            "parameters": [{
                "name": "p",
                "line_key": "1",
                "column_index": 1,
                "initial_value": 0.0,
                "min_value": 0.0,
                "max_value": 2.0,
                "step": 0.1,
            }],
            "objectives": [
                {"name": "left", "value_column_index": 1, "reference_value": 0.0},
                {"name": "right", "value_column_index": 2, "reference_value": 0.0},
            ],
            "integral_objective": {
                "time_column_index": 0,
                "csv_relative_path": "dummy.csv",
            },
            "integral_optimizer": {
                "optimizer": "genetic",
                "max_function_evaluations": 8,
                "max_relap_retries": 0,
                "random_seed": 9,
                "ga_crossover_probability": 0.9,
                "ga_mutation_scale": 0.1,
            },
        }

        with mock.patch("auto_optimize._try_relap_with_recovery", side_effect=fake_try_relap), \
             mock.patch("auto_optimize.refresh_live_csv", return_value={
                 "success": True,
                 "csv_path": __file__,
                 "rows": 2,
             }), \
             mock.patch("auto_optimize.compute_weighted_objective", side_effect=fake_weighted_objective), \
             mock.patch("auto_optimize.load_indta", return_value=[]), \
             mock.patch("auto_optimize.update_parameters_in_indta", return_value=[]), \
             mock.patch("auto_optimize.save_indta"):
            result = run_optimization_minimize_time_integral(
                env="dev",
                config_override=cfg,
                on_log=lambda _msg: None,
            )

        self.assertTrue(result["success"])
        self.assertIn("pareto_front", result)
        self.assertTrue(result["pareto_front"])
        self.assertEqual(len(result["final_objective_values"]), 2)
        self.assertLessEqual(len(result["history"]), 8)
        self.assertAlmostEqual(result["final_parameter_values"][0], 1.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
