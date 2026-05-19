"""单元测试：auto_optimize._compute_recovery_candidate（不依赖 RELAP）。"""

from __future__ import annotations

import unittest

from auto_optimize import _compute_recovery_candidate


class TestRecoveryCandidate(unittest.TestCase):
    def test_prefers_last_safe_when_available(self) -> None:
        # last_safe=20 优先于 initial=100 与中点=250
        new = _compute_recovery_candidate(
            failed_x=200.0, last_safe_x=20.0, initial_value=100.0, bracket=(1.0, 500.0)
        )
        self.assertAlmostEqual(new, (200.0 + 20.0) / 2.0)

    def test_falls_back_to_initial_when_no_last_safe(self) -> None:
        new = _compute_recovery_candidate(
            failed_x=200.0, last_safe_x=None, initial_value=10.0, bracket=(1.0, 500.0)
        )
        self.assertAlmostEqual(new, (200.0 + 10.0) / 2.0)

    def test_falls_back_to_bracket_mid_when_initial_equals_failed(self) -> None:
        # 用户场景：initial_value=10 自身失败，没有 last_safe，必须能向中点回退
        new = _compute_recovery_candidate(
            failed_x=10.0, last_safe_x=None, initial_value=10.0, bracket=(1.0, 500.0)
        )
        # 中点 = 250.5；新建议 = (10 + 250.5)/2 = 130.25
        self.assertAlmostEqual(new, (10.0 + 250.5) / 2.0)

    def test_returns_none_when_failed_equals_all_anchors(self) -> None:
        new = _compute_recovery_candidate(
            failed_x=50.0, last_safe_x=None, initial_value=50.0, bracket=(50.0, 50.0)
        )
        self.assertIsNone(new)

    def test_clamps_to_bracket(self) -> None:
        # last_safe 被故意放在区间外；二分结果会越界，必须 clamp 回 bracket
        new = _compute_recovery_candidate(
            failed_x=10.0, last_safe_x=-1000.0, initial_value=5.0, bracket=(1.0, 500.0)
        )
        self.assertGreaterEqual(new, 1.0)
        self.assertLessEqual(new, 500.0)

    def test_skips_anchor_equal_to_failed(self) -> None:
        # last_safe 与 failed_x 重合应被跳过，落到 initial_value
        new = _compute_recovery_candidate(
            failed_x=100.0, last_safe_x=100.0, initial_value=10.0, bracket=(1.0, 500.0)
        )
        self.assertAlmostEqual(new, (100.0 + 10.0) / 2.0)


if __name__ == "__main__":
    unittest.main()
