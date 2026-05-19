"""单元测试：optimization_trace_view（不依赖 RELAP / Streamlit）。"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from optimization_trace_view import (
    abs_dev_integral,
    build_convergence_figure,
    build_parameter_pair_objective_heatmap,
    build_search_axis_figure,
    build_trace_compare_figure,
    eval_id_of,
    flatten_pareto_front,
    format_parameter_vector,
    make_run_snapshot_dir,
    objective_of,
    read_csv_two_columns,
    snapshot_eval_csv,
    _extract_axis_pair_samples,
    _idw_grid_values,
)


def _write_csv(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("time_0,p\n")
        for t, p in rows:
            f.write(f"{t},{p}\n")


class TestSnapshotUtils(unittest.TestCase):
    def test_make_dir_unique_and_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = make_run_snapshot_dir(tmp, "dev")
            self.assertTrue(os.path.isdir(d))
            self.assertIn("runs", d.replace("\\", "/"))

    def test_snapshot_copies_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "stripf.csv")
            _write_csv(src, [(0.0, 1.0), (1.0, 2.0)])
            d = make_run_snapshot_dir(tmp, "dev")
            dst = snapshot_eval_csv(src, d, 5)
            self.assertIsNotNone(dst)
            self.assertTrue(os.path.isfile(dst))
            self.assertTrue(dst.endswith("eval_005.csv"))

    def test_snapshot_missing_source_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = make_run_snapshot_dir(tmp, "dev")
            self.assertIsNone(snapshot_eval_csv(os.path.join(tmp, "missing.csv"), d, 1))


class TestRecordAdapters(unittest.TestCase):
    def test_eval_id_evaluation_then_iteration(self) -> None:
        self.assertEqual(eval_id_of({"evaluation": 7}), 7)
        self.assertEqual(eval_id_of({"iteration": 3}), 3)
        self.assertEqual(eval_id_of({}), 0)

    def test_objective_integral_mode(self) -> None:
        self.assertAlmostEqual(objective_of({"objective_value": 12.5}), 12.5)

    def test_objective_target_band_uses_abs_error(self) -> None:
        self.assertAlmostEqual(objective_of({"error": -0.4}), 0.4)

    def test_format_parameter_vector_prefers_vector(self) -> None:
        self.assertEqual(
            format_parameter_vector({"parameter_values": [1.0, 2.5]}),
            "[1, 2.5]",
        )
        self.assertEqual(format_parameter_vector({"parameter_value": 3.0}), "[3]")

    def test_flatten_pareto_front_sorts_and_expands_columns(self) -> None:
        rows = flatten_pareto_front([
            {
                "evaluation": 2,
                "objective_value": 4.0,
                "parameter_values": [20.0, 30.0],
                "objective_values": [1.0, 3.0],
            },
            {
                "evaluation": 1,
                "objective_value": 2.0,
                "parameter_values": [10.0, 15.0],
                "objective_values": [1.5, 0.5],
            },
        ])
        self.assertEqual([r["evaluation"] for r in rows], [1, 2])
        self.assertEqual(rows[0]["param_1"], 10.0)
        self.assertEqual(rows[0]["param_2"], 15.0)
        self.assertEqual(rows[0]["objective_1"], 1.5)
        self.assertEqual(rows[0]["objective_2"], 0.5)


class TestCsvAndIntegral(unittest.TestCase):
    def test_read_two_columns_skips_bad(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.csv")
            with open(p, "w", encoding="utf-8") as f:
                f.write("t,v\n0,1\nbad,2\n1,3\n")
            t, v = read_csv_two_columns(p, 0, 1)
            self.assertEqual(t, [0.0, 1.0])
            self.assertEqual(v, [1.0, 3.0])

    def test_read_two_columns_out_of_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.csv")
            _write_csv(p, [(0.0, 1.0), (1.0, 2.0)])
            self.assertEqual(read_csv_two_columns(p, 0, 99), ([], []))

    def test_abs_dev_integral_simple(self) -> None:
        # |v − ref| 在 [0,2] 上为 |t − 1|, 积分 = 1
        ts = [0.0, 1.0, 2.0]
        vs = [0.0, 1.0, 2.0]
        self.assertAlmostEqual(abs_dev_integral(ts, vs, 1.0), 1.0)


class TestFigureBuilders(unittest.TestCase):
    """图构建器只验证「不抛、返回 plotly Figure、含必要 trace 数」。"""

    def test_trace_compare_empty(self) -> None:
        fig = build_trace_compare_figure([], {}, 0, 1, 0.0)
        self.assertEqual(len(fig.data), 0)

    def test_trace_compare_with_focus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv1 = os.path.join(tmp, "a.csv")
            csv2 = os.path.join(tmp, "b.csv")
            _write_csv(csv1, [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)])
            _write_csv(csv2, [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)])
            history = [
                {"evaluation": 1, "parameter_value": 10.0, "objective_value": 3.0},
                {"evaluation": 2, "parameter_value": 20.0, "objective_value": 0.5},
            ]
            snapshots = {1: csv1, 2: csv2}
            fig = build_trace_compare_figure(
                history, snapshots, 0, 1, 1.0,
                selected_eval_ids=[1, 2], focus_eval_id=2,
            )
            # 2 条曲线 + focus 的两条透明 fill 线 + ref 线 = 5 trace
            self.assertGreaterEqual(len(fig.data), 4)

    def test_search_axis_empty(self) -> None:
        fig = build_search_axis_figure([], (0.0, 1.0))
        self.assertEqual(len(fig.data), 0)

    def test_search_axis_with_points(self) -> None:
        history = [
            {"evaluation": 1, "parameter_value": 100.0, "objective_value": 10.0},
            {"evaluation": 2, "parameter_value": 250.0, "objective_value": 4.0},
            {"evaluation": 3, "parameter_value": 200.0, "objective_value": 2.0},
        ]
        fig = build_search_axis_figure(history, (1.0, 500.0))
        self.assertGreaterEqual(len(fig.data), 2)

    def test_convergence_with_target_band(self) -> None:
        history = [
            {"iteration": 1, "parameter_value": 100.0, "error": 0.5,
             "criterion_value": 0.45, "success": False},
            {"iteration": 2, "parameter_value": 105.0, "error": -0.2,
             "criterion_value": 0.97, "success": False},
            {"iteration": 3, "parameter_value": 103.0, "error": 0.05,
             "criterion_value": 0.96, "success": True},
        ]
        fig = build_convergence_figure(history)
        self.assertEqual(len(fig.data), 2)
        cum_best_y = list(fig.data[1].y)
        # cumulative best monotone non-increasing
        for i in range(1, len(cum_best_y)):
            self.assertLessEqual(cum_best_y[i], cum_best_y[i - 1])


class TestParameterPairHeatmap(unittest.TestCase):
    def test_idw_matches_at_sample(self) -> None:
        xs = np.array([0.5])
        ys = np.array([0.25])
        fs = np.array([3.0])
        gx = np.linspace(0.0, 1.0, 5)
        gy = np.linspace(0.0, 1.0, 5)
        z = _idw_grid_values(xs, ys, fs, gx, gy)
        iy = int(np.argmin(np.abs(gy - 0.25)))
        ix = int(np.argmin(np.abs(gx - 0.5)))
        self.assertAlmostEqual(float(z[iy, ix]), 3.0, places=5)

    def test_extract_pair_requires_vector(self) -> None:
        hist = [
            {"evaluation": 1, "parameter_value": 1.0, "objective_value": 2.0},
        ]
        xi, xj, f, _ = _extract_axis_pair_samples(hist, 0, 1)
        self.assertEqual(len(f), 0)
        hist2 = [
            {"evaluation": 1, "parameter_values": [0.1, 0.2], "objective_value": 2.0},
        ]
        xi, xj, f, eids = _extract_axis_pair_samples(hist2, 0, 1)
        self.assertEqual(len(f), 1)
        self.assertAlmostEqual(float(xi[0]), 0.1)
        self.assertAlmostEqual(float(xj[0]), 0.2)
        self.assertEqual(int(eids[0]), 1)

    def test_build_pair_heatmap_multi_points(self) -> None:
        hist = [
            {"evaluation": k, "parameter_values": [float(k), float(k) * 0.5],
             "objective_value": float(k * k)}
            for k in range(1, 6)
        ]
        fig = build_parameter_pair_objective_heatmap(
            hist,
            ["p0", "p1"],
            [(0.0, 10.0), (0.0, 10.0)],
            axis_i=0,
            axis_j=1,
            grid_resolution=24,
        )
        self.assertGreaterEqual(len(fig.data), 3)

    def test_build_pair_invalid_axes(self) -> None:
        fig = build_parameter_pair_objective_heatmap(
            [],
            ["a", "b"],
            [(0.0, 1.0), (0.0, 1.0)],
            axis_i=0,
            axis_j=0,
        )
        self.assertEqual(len(fig.data), 0)


if __name__ == "__main__":
    unittest.main()
