"""Unit tests for genetic_optimizer (no RELAP executable required)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from genetic_optimizer import ga_minimize_pareto


class TestGeneticOptimizer(unittest.TestCase):
    def test_single_objective_quadratic_uses_budget_and_finds_best(self) -> None:
        calls = {"n": 0}

        def evaluate(x: np.ndarray):
            calls["n"] += 1
            obj = float((x[0] - 2.0) ** 2)
            return {
                "evaluation": calls["n"],
                "parameter_values": x.tolist(),
                "objective_values": [obj],
                "objective_value": obj,
                "relap_failed": False,
            }

        res = ga_minimize_pareto(
            evaluate,
            [(0.0, 4.0)],
            max_evaluations=20,
            random_seed=7,
        )

        self.assertEqual(calls["n"], 20)
        self.assertEqual(len(res["evaluations"]), 20)
        self.assertTrue(res["success"])
        best = res["best_record"]
        self.assertIsNotNone(best)
        self.assertLess(abs(best["parameter_values"][0] - 2.0), 0.25)
        self.assertLess(best["objective_value"], 0.1)

    def test_two_objective_conflict_returns_pareto_front(self) -> None:
        calls = {"n": 0}

        def evaluate(x: np.ndarray):
            calls["n"] += 1
            f1 = float(x[0] ** 2)
            f2 = float((x[0] - 2.0) ** 2)
            return {
                "evaluation": calls["n"],
                "parameter_values": x.tolist(),
                "objective_values": [f1, f2],
                "objective_value": f1 + f2,
                "relap_failed": False,
            }

        res = ga_minimize_pareto(
            evaluate,
            [(0.0, 2.0)],
            max_evaluations=18,
            random_seed=11,
        )

        front = res["pareto_front"]
        self.assertGreaterEqual(len(front), 3)
        self.assertTrue(all(len(row["objective_values"]) == 2 for row in front))
        xs = [row["parameter_values"][0] for row in front]
        self.assertGreaterEqual(min(xs), 0.0)
        self.assertLessEqual(max(xs), 2.0)

    def test_max_evaluations_is_hard_limit(self) -> None:
        calls = {"n": 0}

        def evaluate(x: np.ndarray):
            calls["n"] += 1
            obj = float(x[0] ** 2 + x[1] ** 2)
            return {
                "evaluation": calls["n"],
                "parameter_values": x.tolist(),
                "objective_values": [obj],
                "objective_value": obj,
                "relap_failed": False,
            }

        res = ga_minimize_pareto(
            evaluate,
            [(-1.0, 1.0), (-1.0, 1.0)],
            max_evaluations=7,
            random_seed=3,
        )
        self.assertEqual(calls["n"], 7)
        self.assertEqual(len(res["evaluations"]), 7)

    def test_failed_points_do_not_enter_pareto_front(self) -> None:
        calls = {"n": 0}

        def evaluate(x: np.ndarray):
            calls["n"] += 1
            failed = bool(x[0] < 0.5)
            obj = float("inf") if failed else float((x[0] - 1.0) ** 2)
            return {
                "evaluation": calls["n"],
                "parameter_values": x.tolist(),
                "objective_values": [obj],
                "objective_value": obj,
                "relap_failed": failed,
            }

        res = ga_minimize_pareto(
            evaluate,
            [(0.0, 2.0)],
            max_evaluations=14,
            random_seed=5,
        )
        self.assertTrue(res["pareto_front"])
        self.assertTrue(
            all(row["parameter_values"][0] >= 0.5 for row in res["pareto_front"])
        )
        self.assertTrue(
            all(math.isfinite(row["objective_value"]) for row in res["pareto_front"])
        )

    def test_all_candidates_stay_inside_bounds(self) -> None:
        seen = []

        def evaluate(x: np.ndarray):
            seen.append(x.copy())
            obj = float(np.sum(x * x))
            return {
                "parameter_values": x.tolist(),
                "objective_values": [obj],
                "objective_value": obj,
                "relap_failed": False,
            }

        ga_minimize_pareto(
            evaluate,
            [(-2.0, -1.0), (3.0, 4.0)],
            max_evaluations=16,
            random_seed=19,
            mutation_scale=1.5,
        )
        arr = np.asarray(seen)
        self.assertTrue(np.all(arr[:, 0] >= -2.0))
        self.assertTrue(np.all(arr[:, 0] <= -1.0))
        self.assertTrue(np.all(arr[:, 1] >= 3.0))
        self.assertTrue(np.all(arr[:, 1] <= 4.0))


if __name__ == "__main__":
    unittest.main()

