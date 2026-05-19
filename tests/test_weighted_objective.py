"""单元测试：integral_objective.compute_weighted_objective（不依赖 RELAP）。"""

from __future__ import annotations

import os
import tempfile
import unittest

from integral_objective import compute_weighted_objective


def _write_csv(path: str, header: str, rows: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(rows)


class TestWeightedObjective(unittest.TestCase):
    def test_single_objective_matches_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.csv")
            # t=[0,1], v=[1, 3]; ref=2 → |Δ|=[1,1] → trapezoid integral = 1.0
            _write_csv(p, "t,v", "0,1\n1,3\n")
            agg, parts = compute_weighted_objective(
                p, time_column_index=0,
                objectives=[
                    {"value_column_index": 1, "reference_value": 2.0,
                     "deviation": "absolute", "weight": 1.0, "scale": 1.0},
                ],
            )
            self.assertAlmostEqual(agg, 1.0, places=6)
            self.assertEqual(len(parts), 1)
            self.assertAlmostEqual(parts[0], 1.0, places=6)

    def test_two_objectives_sum_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.csv")
            # t=[0,1], v1=[1,3] (ref=2) → integral 1
            #         v2=[10,12] (ref=11) → integral 1
            _write_csv(p, "t,v1,v2", "0,1,10\n1,3,12\n")
            agg, parts = compute_weighted_objective(
                p, time_column_index=0,
                objectives=[
                    {"value_column_index": 1, "reference_value": 2.0,
                     "weight": 1.0, "scale": 1.0},
                    {"value_column_index": 2, "reference_value": 11.0,
                     "weight": 2.0, "scale": 1.0},
                ],
            )
            # aggregate = 1 * 1 + 2 * 1 = 3
            self.assertAlmostEqual(agg, 3.0, places=6)
            self.assertAlmostEqual(parts[0], 1.0)
            self.assertAlmostEqual(parts[1], 1.0)

    def test_scale_normalizes_magnitudes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.csv")
            # 设计两条同形状但量级差 1e6 倍的曲线：
            #   v1 偏差积分 = 1e6
            #   v2 偏差积分 = 1
            _write_csv(p, "t,v1,v2", "0,1000000,1\n1,3000000,3\n")
            # ref1=2e6 → integral=1e6；ref2=2 → integral=1
            agg, parts = compute_weighted_objective(
                p, time_column_index=0,
                objectives=[
                    {"value_column_index": 1, "reference_value": 2e6,
                     "weight": 1.0, "scale": 1e6},  # 归一化后 = 1
                    {"value_column_index": 2, "reference_value": 2.0,
                     "weight": 1.0, "scale": 1.0},  # = 1
                ],
            )
            self.assertAlmostEqual(parts[0], 1e6, places=2)
            self.assertAlmostEqual(parts[1], 1.0, places=6)
            # 经 scale 归一化后两个目标贡献相等；总和 = 2
            self.assertAlmostEqual(agg, 2.0, places=6)

    def test_empty_objectives_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.csv")
            _write_csv(p, "t,v", "0,0\n1,1\n")
            with self.assertRaises(ValueError):
                compute_weighted_objective(p, 0, [])

    def test_invalid_deviation_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.csv")
            _write_csv(p, "t,v", "0,0\n1,1\n")
            with self.assertRaises(ValueError):
                compute_weighted_objective(
                    p, 0,
                    [{"value_column_index": 1, "reference_value": 0.0,
                      "deviation": "bogus"}],
                )

    def test_zero_scale_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.csv")
            _write_csv(p, "t,v", "0,0\n1,1\n")
            with self.assertRaises(ValueError):
                compute_weighted_objective(
                    p, 0,
                    [{"value_column_index": 1, "reference_value": 0.0,
                      "weight": 1.0, "scale": 0.0}],
                )


if __name__ == "__main__":
    unittest.main()
