"""单元测试：optimization_trace_view.extract_integrals / build_integral_bar_figure。"""

from __future__ import annotations

import math
import unittest

from optimization_trace_view import (
    build_integral_bar_figure,
    extract_integrals,
)


class TestExtractIntegrals(unittest.TestCase):
    def test_empty_history(self) -> None:
        self.assertEqual(extract_integrals([]), [])

    def test_only_integral_mode_records_kept(self) -> None:
        h = [
            # target_band 模式记录（无 objective_value，仅 error / criterion_value）
            {"iteration": 1, "parameter_value": 5.0,
             "error": 0.1, "criterion_value": 1.0},
            # 积分模式成功
            {"evaluation": 2, "parameter_value": 10.0,
             "objective_value": 1.5e6},
        ]
        rows = extract_integrals(h)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["eval_id"], 2)
        self.assertAlmostEqual(rows[0]["integral"], 1.5e6)

    def test_skips_failed_and_inf_records(self) -> None:
        h = [
            {"evaluation": 1, "parameter_value": 10.0, "objective_value": 5.0},
            {"evaluation": 2, "parameter_value": 999.0,
             "objective_value": float("inf"), "relap_failed": True},
            {"evaluation": 3, "parameter_value": 50.0,
             "objective_value": float("nan")},
            {"evaluation": 4, "parameter_value": 30.0,
             "objective_value": 4.0, "post_run_error": "extract failed"},
            {"evaluation": 5, "parameter_value": 25.0, "objective_value": 3.0},
        ]
        rows = extract_integrals(h)
        ids = [r["eval_id"] for r in rows]
        self.assertEqual(ids, [1, 5])

    def test_is_best_marks_running_minimum(self) -> None:
        h = [
            {"evaluation": 1, "parameter_value": 10.0, "objective_value": 5.0},
            {"evaluation": 2, "parameter_value": 20.0, "objective_value": 3.0},
            {"evaluation": 3, "parameter_value": 30.0, "objective_value": 4.0},
            {"evaluation": 4, "parameter_value": 40.0, "objective_value": 2.0},
            {"evaluation": 5, "parameter_value": 50.0, "objective_value": 4.5},
        ]
        rows = extract_integrals(h)
        self.assertEqual([r["is_best"] for r in rows], [True, True, False, True, False])

    def test_carries_retries_field(self) -> None:
        h = [
            {"evaluation": 1, "parameter_value": 10.0,
             "objective_value": 1.0, "retries": 3},
            {"evaluation": 2, "parameter_value": 20.0,
             "objective_value": 2.0},  # missing retries → 0
        ]
        rows = extract_integrals(h)
        self.assertEqual(rows[0]["retries"], 3)
        self.assertEqual(rows[1]["retries"], 0)


class TestIntegralBarFigure(unittest.TestCase):
    def test_empty_history_returns_placeholder(self) -> None:
        fig = build_integral_bar_figure([])
        self.assertEqual(len(fig.data), 0)
        ann_text = " ".join(a.text for a in (fig.layout.annotations or []))
        self.assertIn("尚未捕获", ann_text)

    def test_with_records_has_bar_trace(self) -> None:
        h = [
            {"evaluation": 1, "parameter_value": 10.0, "objective_value": 5.0},
            {"evaluation": 2, "parameter_value": 20.0, "objective_value": 3.0},
        ]
        fig = build_integral_bar_figure(h)
        self.assertGreaterEqual(len(fig.data), 1)
        bar = fig.data[0]
        self.assertEqual(list(bar.x), [1, 2])
        self.assertEqual(list(bar.y), [5.0, 3.0])
        # 两条都应被标为 best（递减序列）→ 两根都是绿色
        self.assertEqual(list(bar.marker.color), ["#27ae60", "#27ae60"])

    def test_color_distinguishes_non_best(self) -> None:
        h = [
            {"evaluation": 1, "parameter_value": 10.0, "objective_value": 3.0},
            {"evaluation": 2, "parameter_value": 20.0, "objective_value": 5.0},
        ]
        fig = build_integral_bar_figure(h)
        bar = fig.data[0]
        # eval 1=最优(绿)，eval 2=未刷新(灰)
        colors = list(bar.marker.color)
        self.assertEqual(colors[0], "#27ae60")
        self.assertNotEqual(colors[1], "#27ae60")


if __name__ == "__main__":
    unittest.main()
