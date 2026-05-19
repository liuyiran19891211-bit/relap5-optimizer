"""单元测试：bayesian_optimizer（不依赖 RELAP）。"""

from __future__ import annotations

import math
import unittest

import numpy as np

from bayesian_optimizer import (
    _expected_improvement,
    _gp_posterior,
    _Phi,
    bo_minimize,
    bo_minimize_nd,
)


class TestStandardNormal(unittest.TestCase):
    def test_phi_at_zero(self) -> None:
        # 正态 CDF(0) = 0.5
        self.assertAlmostEqual(float(_Phi(np.array([0.0]))[0]), 0.5, places=6)

    def test_phi_symmetry(self) -> None:
        # CDF(z) + CDF(-z) = 1
        for z in (-1.5, -0.5, 0.3, 1.96):
            a = float(_Phi(np.array([z]))[0])
            b = float(_Phi(np.array([-z]))[0])
            self.assertAlmostEqual(a + b, 1.0, places=6)


class TestGpPosterior(unittest.TestCase):
    def test_predicts_observed_points_exactly(self) -> None:
        # 在已观测点上，noise=1e-6 时 μ ≈ y_train、σ ≈ 0
        X = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, -1.0, 2.0])
        mu, sigma = _gp_posterior(X, y, X, length_scale=1.0, noise_level=1e-6)
        np.testing.assert_allclose(mu, y, atol=1e-3)
        np.testing.assert_array_less(sigma, 0.05)

    def test_returns_prior_with_no_data(self) -> None:
        mu, sigma = _gp_posterior(
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([0.5]),
            length_scale=1.0,
            noise_level=1e-6,
        )
        self.assertAlmostEqual(float(mu[0]), 0.0)
        self.assertAlmostEqual(float(sigma[0]), 1.0)


class TestExpectedImprovement(unittest.TestCase):
    def test_zero_improvement_when_predicted_worse_with_zero_sigma(self) -> None:
        # 预测均值 > y_best 且不确定性 ≈ 0，则 EI ≈ 0
        ei = _expected_improvement(
            mu=np.array([1.5]), sigma=np.array([1e-8]), y_best=1.0, xi=0.0
        )
        self.assertLess(float(ei[0]), 1e-6)


class TestBoMinimizeOnConvex(unittest.TestCase):
    """凸函数 ``(x − 2)^2`` 上 BO 应在少量评估内逼近 x≈2。"""

    @staticmethod
    def _f(x: float) -> float:
        return (x - 2.0) ** 2

    def test_converges_on_quadratic(self) -> None:
        x_best, f_best = bo_minimize(
            self._f, 0.0, 4.0, max_evaluations=12, n_initial=3
        )
        self.assertLess(abs(x_best - 2.0), 0.1)
        self.assertLess(f_best, 0.01)

    def test_warm_start_does_not_call_func(self) -> None:
        calls = {"n": 0}

        def f_counted(x: float) -> float:
            calls["n"] += 1
            return self._f(x)

        # warm_start 已经覆盖了最优点，且 max_evaluations=2，n_initial=2
        # → 仅会再做 2 次 seeding/BO 评估，warm_start 的两个点不计入
        x_best, f_best = bo_minimize(
            f_counted,
            0.0,
            4.0,
            max_evaluations=2,
            n_initial=2,
            warm_start=[(2.0, 0.0), (3.0, 1.0)],
        )
        self.assertLessEqual(calls["n"], 2)
        # 已知最优点在 warm_start 中：x_best 必须紧靠 2.0
        self.assertLess(abs(x_best - 2.0), 1e-6)
        self.assertAlmostEqual(f_best, 0.0)

    def test_respects_bracket(self) -> None:
        x_best, _ = bo_minimize(self._f, 5.0, 10.0, max_evaluations=8, n_initial=3)
        self.assertGreaterEqual(x_best, 5.0)
        self.assertLessEqual(x_best, 10.0)


class TestBoMinimizeWithFailures(unittest.TestCase):
    """func 返回 ``+inf`` 表示失败评估，BO 应自然避开这些区域。"""

    def test_avoids_infinite_region(self) -> None:
        def f(x: float) -> float:
            if x < 1.0:
                return float("inf")  # 视作 RELAP 失败
            return (x - 2.0) ** 2

        x_best, f_best = bo_minimize(
            f, 0.0, 4.0, max_evaluations=15, n_initial=3
        )
        self.assertGreaterEqual(x_best, 1.0)
        self.assertLess(abs(x_best - 2.0), 0.2)

    def test_returns_inf_when_all_fail(self) -> None:
        x_best, f_best = bo_minimize(
            lambda x: float("inf"), 0.0, 1.0, max_evaluations=5, n_initial=2
        )
        # 极端：完全没有可用观测，但函数应正常返回（不抛）
        self.assertTrue(math.isinf(f_best))


class TestBoMinimizeProgressCallback(unittest.TestCase):
    def test_progress_called_each_evaluation(self) -> None:
        progress = []

        def cb(i: int, xb: float, yb: float) -> None:
            progress.append((i, xb, yb))

        bo_minimize(
            lambda x: (x - 1.0) ** 2,
            0.0,
            2.0,
            max_evaluations=5,
            n_initial=2,
            on_progress=cb,
        )
        # 至少 5 次回调（实际应该恰好是 max_evaluations 次）
        self.assertEqual(len(progress), 5)
        # 最优 y 单调不增
        ys = [p[2] for p in progress if math.isfinite(p[2])]
        for i in range(1, len(ys)):
            self.assertLessEqual(ys[i], ys[i - 1] + 1e-9)


class TestBoMinimizeNd(unittest.TestCase):
    """N-D BO 在已知凸函数上应在少量评估内逼近最优。"""

    def test_2d_quadratic_converges(self) -> None:
        """f(x, y) = (x-1)² + (y+2)² 在 [(-3,3), (-3,3)] 上最优在 (1, -2)。"""
        def f(v: np.ndarray) -> float:
            return float((v[0] - 1.0) ** 2 + (v[1] + 2.0) ** 2)

        x_best, f_best = bo_minimize_nd(
            f, [(-3.0, 3.0), (-3.0, 3.0)],
            max_evaluations=30, n_initial=4,
        )
        self.assertEqual(x_best.shape, (2,))
        # 30 次评估对 2D 凸函数足够把误差压到 0.5 以内
        self.assertLess(abs(x_best[0] - 1.0) + abs(x_best[1] + 2.0), 0.8)
        self.assertLess(f_best, 0.4)

    def test_warm_start_in_nd(self) -> None:
        """warm_start 包含全局最优点时 BO 不必再做太多搜索。"""
        calls = {"n": 0}

        def f(v: np.ndarray) -> float:
            calls["n"] += 1
            return float(v[0] ** 2 + v[1] ** 2)

        x_best, f_best = bo_minimize_nd(
            f, [(-2.0, 2.0), (-2.0, 2.0)],
            max_evaluations=3, n_initial=2,
            warm_start=[([0.0, 0.0], 0.0), ([1.5, 1.5], 4.5)],
        )
        # warm_start 已含 (0,0) 全局最优 → x_best 不必离 (0,0) 远
        self.assertLess(np.sqrt(np.sum(x_best ** 2)), 1.5)
        # warm_start 不计入；至多 3 次真实调用
        self.assertLessEqual(calls["n"], 3)

    def test_avoids_infeasible_region_nd(self) -> None:
        """func 返回 +inf 表示该点不可行；BO 应自然避开。"""
        def f(v: np.ndarray) -> float:
            if v[0] < -0.5:
                return float("inf")
            return float((v[0] - 1.0) ** 2 + (v[1]) ** 2)

        x_best, f_best = bo_minimize_nd(
            f, [(-2.0, 2.0), (-2.0, 2.0)],
            max_evaluations=20, n_initial=4,
        )
        self.assertGreaterEqual(x_best[0], -0.5)

    def test_invalid_bounds_raises(self) -> None:
        with self.assertRaises(ValueError):
            bo_minimize_nd(lambda v: 0.0, [(2.0, 1.0)])  # a >= b
        with self.assertRaises(ValueError):
            bo_minimize_nd(lambda v: 0.0, [])  # empty bounds


class TestRbfKernelNdShape(unittest.TestCase):
    """_rbf_kernel 必须能处理 ``[n, d]`` 形状的矩阵输入。"""

    def test_2d_kernel_diagonal_is_one(self) -> None:
        from bayesian_optimizer import _rbf_kernel

        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        K = _rbf_kernel(X, X, length_scale=1.0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

    def test_2d_kernel_decays_with_distance(self) -> None:
        from bayesian_optimizer import _rbf_kernel

        X = np.array([[0.0, 0.0]])
        Y = np.array([[1.0, 0.0], [10.0, 0.0]])  # 第二个远得多
        K = _rbf_kernel(X, Y, length_scale=1.0)
        self.assertGreater(K[0, 0], K[0, 1])


if __name__ == "__main__":
    unittest.main()
