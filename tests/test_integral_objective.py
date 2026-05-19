"""Unit tests for integral_objective (no RELAP executables required)."""

import os
import tempfile
import unittest

from integral_objective import (
    compute_time_integral_objective,
    golden_section_minimize,
    select_best_value_column,
    trapezoid_integral,
)


class TestTrapezoid(unittest.TestCase):
    def test_line(self) -> None:
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        self.assertAlmostEqual(trapezoid_integral(x, y), 2.0)

    def test_constant_zero_deviation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("t,c0,c1\n")
                f.write("0,0,1.55e7\n")
                f.write("1,0,1.55e7\n")
            obj = compute_time_integral_objective(
                path, 0, 2, 15.5e6, deviation="absolute"
            )
            self.assertAlmostEqual(obj, 0.0, places=6)

    def test_signed_linear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("t,v\n")
                f.write("0,0\n")
                f.write("1,2\n")
            # ∫_0^1 2t dt = 1 when v = 2t, col1 is index 1; ref 0 => signed integrand 2t
            obj = compute_time_integral_objective(
                path, 0, 1, 0.0, deviation="signed"
            )
            self.assertAlmostEqual(obj, 1.0, places=6)


class TestSelectBestValueColumn(unittest.TestCase):
    """value_column_index 越界时按 reference_value 量级匹配自动选列。"""

    def _write(self, tmp: str, header: str, row1: str) -> str:
        path = os.path.join(tmp, "x.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write(row1 + "\n")
        return path

    def test_in_range_returns_configured_no_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, "t,a,b,c", "0,1.0,2.0,3.0")
            col, warn = select_best_value_column(
                p, time_col=0, reference_value=2.0, configured_col=2
            )
            self.assertEqual(col, 2)
            self.assertIsNone(warn)

    def test_out_of_range_picks_log_closest_to_reference(self) -> None:
        """用户场景：CSV 13 列、配置 16；ref=1.55e7 → 应选首行 ~1.57e7 的列。"""
        with tempfile.TemporaryDirectory() as tmp:
            header = (
                "time_0,rktpow_0,tempf_a,tempf_b,tempf_c,tempf_d,p_pressure,"
                "mflowj_0,mflowj_1,mflowj_2,mflowj_3,mflowj_4,mflowj_5"
            )
            row1 = (
                "0.0,3.6e9,582.7,549.8,582.7,549.8,1.572280e+07,"
                "1.38e4,4.61e3,1.32e3,1.32e3,4.41e2,4.41e2"
            )
            p = self._write(tmp, header, row1)
            col, warn = select_best_value_column(
                p, time_col=0, reference_value=15500000.0, configured_col=16
            )
            self.assertEqual(col, 6)  # p_pressure ≈ 1.57e7
            self.assertIsNotNone(warn)
            self.assertIn("p_pressure", warn)
            self.assertIn("16", warn)

    def test_skips_time_column(self) -> None:
        """time_col 即使数值最贴近也不会被选。"""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, "t,v", "1.0,500.0")
            col, warn = select_best_value_column(
                p, time_col=0, reference_value=1.0, configured_col=99
            )
            self.assertEqual(col, 1)  # not 0 (time)
            self.assertIsNotNone(warn)

    def test_no_data_row_falls_back_to_last_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("a,b,c\n")  # header only
            col, warn = select_best_value_column(
                path, time_col=0, reference_value=1.0, configured_col=99
            )
            self.assertEqual(col, 2)  # last column
            self.assertIn("无数据行", warn)

    def test_unreadable_path(self) -> None:
        col, warn = select_best_value_column(
            "/no/such/file.csv", time_col=0, reference_value=1.0, configured_col=5
        )
        self.assertEqual(col, 5)  # 退回 configured，不抛
        self.assertIn("无法读取", warn)


class TestGoldenSection(unittest.TestCase):
    def test_quadratic(self) -> None:
        def f(x: float) -> float:
            return (x - 2.0) ** 2

        xb, fb = golden_section_minimize(f, 0.0, 4.0, tol=1e-9, max_evaluations=60)
        self.assertAlmostEqual(xb, 2.0, places=5)
        self.assertLess(fb, 1e-8)


if __name__ == "__main__":
    unittest.main()
