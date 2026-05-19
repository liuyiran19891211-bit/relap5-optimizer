"""
1D 贝叶斯优化（GP + Expected Improvement），仅依赖 numpy 与 ``math.erf``。

为什么 DIY：项目当前没有 sklearn / scipy / scikit-optimize，按用户规则「优先简单
方案 / 避免引入新技术」，自己用 numpy 写一个 ~130 行的最小可用 BO，签名与
``integral_objective.golden_section_minimize`` 完全一致：

    x_best, f_best = bo_minimize(func, a, b, max_evaluations=N)

可在 ``auto_optimize.run_optimization_minimize_time_integral`` 处插拔切换。

设计要点
---------
- **核**：固定长度尺度的 RBF；length_scale 默认 ``(b-a)/5`` —— 1D 经验值
- **采集函数**：Expected Improvement（最小化场景），带 ``xi`` 探索权
- **失败评估**（func 返回 ``+inf``）：用一个高出当前 y 范围 10 倍的 penalty 喂给 GP，
  让 EI 自然避开那些区域（不需另写约束分类器）
- **Warm start**：可传入已知 ``(x, y)`` 列表免费"白嫖"——典型用法是把 caller 已经
  跑出的 ``initial_value`` 评估结果灌进来，不再重复评
- **数值稳定**：K + ``noise_level * I`` 的 Cholesky 分解
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# ---- 标准正态 CDF / PDF（向量化）----------------------------------------------

def _phi(z: np.ndarray) -> np.ndarray:
    """标准正态 PDF，向量化。"""
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _Phi(z: np.ndarray) -> np.ndarray:
    """标准正态 CDF，用 ``math.erf`` 实现，规模 1024 也只需 ~1ms。"""
    z = np.asarray(z, dtype=float)
    flat = z.flatten()
    out = np.fromiter(
        (0.5 * (1.0 + math.erf(zi / math.sqrt(2.0))) for zi in flat),
        dtype=float,
        count=flat.size,
    )
    return out.reshape(z.shape)


# ---- GP（RBF kernel） --------------------------------------------------------

def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float) -> np.ndarray:
    """
    RBF 核：``K[i,j] = exp(-||X1[i]-X2[j]||² / (2 * length_scale²))``。

    支持两种输入：
      - 1D：``X1, X2`` 形如 ``[n]``（向量），距离即标量差
      - N-D：``X1, X2`` 形如 ``[n, d]``，距离为欧氏距离的平方
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    if X1.ndim == 1 and X2.ndim == 1:
        diff = X1.reshape(-1, 1) - X2.reshape(1, -1)
        sq = diff ** 2
    else:
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)
        # ||a-b||² = |a|² + |b|² - 2 a·b
        a2 = np.sum(X1 * X1, axis=1).reshape(-1, 1)
        b2 = np.sum(X2 * X2, axis=1).reshape(1, -1)
        sq = np.maximum(a2 + b2 - 2.0 * (X1 @ X2.T), 0.0)
    return np.exp(-sq / (2.0 * length_scale * length_scale))


def _gp_posterior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    length_scale: float,
    noise_level: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GP 后验 ``μ(x*), σ(x*)``，支持 1D 与 N-D。

    ``X_train``/``X_query`` 可以是 ``[n]`` (1D) 或 ``[n, d]`` (N-D)。
    用 mean-zero 先验：拟合前减去 ``y_train`` 均值，预测后加回。
    用 Cholesky 分解保证数值稳定（K + ``noise_level * I``）。
    """
    n = X_train.shape[0] if X_train.ndim >= 1 else 0
    n_query = X_query.shape[0] if X_query.ndim >= 1 else 0
    if n == 0:
        return (
            np.zeros(n_query, dtype=float),
            np.ones(n_query, dtype=float),
        )

    mean_y = float(np.mean(y_train))
    yc = y_train - mean_y

    K = _rbf_kernel(X_train, X_train, length_scale) + noise_level * np.eye(n)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))

    K_s = _rbf_kernel(X_train, X_query, length_scale)
    mu = (K_s.T @ alpha) + mean_y

    v = np.linalg.solve(L, K_s)
    var = 1.0 - np.sum(v * v, axis=0)
    var = np.maximum(var, 1e-12)
    sigma = np.sqrt(var)
    return mu.flatten(), sigma.flatten()


def _expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01
) -> np.ndarray:
    """期望改进（最小化场景）：``EI(x) = E[max(y_best - f(x) - xi, 0)]``。"""
    sigma = np.maximum(sigma, 1e-12)
    improvement = y_best - mu - xi
    z = improvement / sigma
    return improvement * _Phi(z) + sigma * _phi(z)


# ---- 主入口 ------------------------------------------------------------------

def bo_minimize_nd(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[Tuple[float, float]],
    *,
    max_evaluations: int = 25,
    warm_start: Optional[Sequence[Tuple[Sequence[float], float]]] = None,
    n_initial: int = 3,
    exploration_xi: float = 0.01,
    candidate_pool: int = 4096,
    length_scale: Optional[float] = None,
    noise_level: float = 1e-6,
    on_progress: Optional[Callable[[int, np.ndarray, float], None]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float]:
    """
    多维（N-D）贝叶斯优化。``func`` 接收 ``ndarray[d]``，返回标量（``+inf`` = 失败）。

    Parameters
    ----------
    bounds : ``[(a₁,b₁), …, (a_d,b_d)]`` —— 每个参数的搜索区间
    warm_start : ``[(x_vec, y), …]`` —— 已知点（不会重复 call func）
    candidate_pool : N-D 时改为在 bounds 内均匀采样 ``candidate_pool`` 个候选点
                     选最大 EI；高维下不再使用 dense grid
    length_scale : 单一各向同性尺度（默认 ``min(b_i-a_i) / 5``）；
                   N-D 时建议用户先把 bounds 归一化到相近量级以保证拟合质量

    Returns
    -------
    ``(x_best: ndarray[d], f_best: float)``
    """
    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError(f"bounds must be List[(a,b)], got shape {bounds_arr.shape}")
    a_vec = bounds_arr[:, 0]
    b_vec = bounds_arr[:, 1]
    if np.any(a_vec >= b_vec):
        raise ValueError(f"Each bound (a,b) must have a<b, got {bounds!r}")
    d = bounds_arr.shape[0]

    if rng is None:
        rng = np.random.default_rng(seed=12345)
    if length_scale is None:
        length_scale = max(float(np.min(b_vec - a_vec)) / 5.0, 1e-9)

    X_obs: List[np.ndarray] = []
    y_obs: List[float] = []
    X_failed: List[np.ndarray] = []
    eval_count = 0

    def _emit_progress() -> None:
        if on_progress is None:
            return
        try:
            if y_obs:
                i = int(np.argmin(y_obs))
                on_progress(eval_count, X_obs[i].copy(), y_obs[i])
            else:
                on_progress(eval_count, np.full(d, np.nan), float("inf"))
        except Exception:
            pass

    def _record_observation(x_eval: np.ndarray, y_eval: float) -> None:
        nonlocal eval_count
        if math.isfinite(y_eval):
            X_obs.append(np.array(x_eval, dtype=float))
            y_obs.append(float(y_eval))
        else:
            X_failed.append(np.array(x_eval, dtype=float))
        eval_count += 1
        _emit_progress()

    # 1) Warm start：不计入 eval_count
    if warm_start:
        for x_w, y_w in warm_start:
            try:
                x_arr = np.asarray(x_w, dtype=float).flatten()
                y_f = float(y_w)
            except (TypeError, ValueError):
                continue
            if x_arr.shape[0] != d:
                continue
            if np.any(x_arr < a_vec) or np.any(x_arr > b_vec):
                continue
            if math.isfinite(y_f):
                X_obs.append(x_arr)
                y_obs.append(y_f)
            else:
                X_failed.append(x_arr)

    # 2) 观测不足以训 GP 时，先在 bounds 内均匀采样补齐
    needed_initial = max(2, int(n_initial)) - len(X_obs)
    if needed_initial > 0 and eval_count < max_evaluations:
        for _ in range(needed_initial):
            if eval_count >= max_evaluations:
                break
            cand = a_vec + rng.random(d) * (b_vec - a_vec)
            cy = func(cand)
            _record_observation(cand, float(cy))

    # 3) 主 BO 循环
    while eval_count < max_evaluations:
        if y_obs:
            ymax = max(y_obs)
            ymin = min(y_obs)
            penalty = ymax + 10.0 * max(ymax - ymin, abs(ymax) + 1.0)
        else:
            penalty = 1.0

        if X_obs or X_failed:
            X_train = np.array(
                [arr for arr in (X_obs + X_failed)], dtype=float
            )
            y_train = np.array(
                y_obs + [penalty] * len(X_failed), dtype=float
            )
        else:
            X_train = np.empty((0, d), dtype=float)
            y_train = np.empty((0,), dtype=float)

        if X_train.shape[0] < 1:
            next_x = a_vec + rng.random(d) * (b_vec - a_vec)
        else:
            # 候选池：均匀采样 N-D 内的随机点
            pool = a_vec + rng.random((int(candidate_pool), d)) * (b_vec - a_vec)
            mu, sigma = _gp_posterior(
                X_train, y_train, pool, length_scale, noise_level
            )
            y_best = min(y_obs) if y_obs else penalty
            ei = _expected_improvement(mu, sigma, y_best, xi=exploration_xi)

            # 抑制贴近已采样点的 EI
            close_r = max(length_scale * 0.05, float(np.min(b_vec - a_vec)) * 1e-4)
            for x_sampled in X_train:
                dist = np.sqrt(np.sum((pool - x_sampled.reshape(1, -1)) ** 2, axis=1))
                ei[dist < close_r] = -np.inf

            if not np.any(np.isfinite(ei)) or float(np.max(ei)) <= -np.inf:
                next_x = a_vec + rng.random(d) * (b_vec - a_vec)
            else:
                next_x = pool[int(np.argmax(ei))].copy()

        next_x = np.maximum(a_vec, np.minimum(b_vec, next_x))
        next_y = func(next_x)
        _record_observation(next_x, float(next_y))

    if y_obs:
        i_best = int(np.argmin(y_obs))
        return X_obs[i_best].copy(), y_obs[i_best]
    return np.full(d, np.nan), float("inf")


def bo_minimize(
    func: Callable[[float], float],
    a: float,
    b: float,
    *,
    max_evaluations: int = 25,
    warm_start: Optional[Sequence[Tuple[float, float]]] = None,
    n_initial: int = 3,
    exploration_xi: float = 0.01,
    candidate_grid: int = 1024,
    length_scale: Optional[float] = None,
    noise_level: float = 1e-6,
    on_progress: Optional[Callable[[int, float, float], None]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    在 ``[a, b]`` 上最小化 ``func``，签名与 :func:`integral_objective.golden_section_minimize`
    兼容。

    Parameters
    ----------
    func
        1D 实值函数；可能返回 ``+inf`` 表示评估失败（BO 会避开该区域）
    warm_start
        已知 ``(x, y)`` 列表 —— BO 用作初值，**不会再调用 func**；
        典型场景是把 caller 已经跑过的 ``initial_value`` 评估直接喂进来
    n_initial
        warm_start 中可用观测不足时，BO 在 ``[a, b]`` 内补几个均匀点（最少 2，
        用于初始化 GP）；这些点会真的调用 func
    exploration_xi
        EI 的探索权，越小越偏 exploitation；典型值 ``0.001 ~ 0.1``
    candidate_grid
        在 ``[a, b]`` 上把 EI 离散化到这么多点上挑 argmax
    length_scale
        RBF 核长度尺度；默认 ``(b-a)/5`` —— 1D 经验值
    noise_level
        加在 K 对角的小抖动，保证 Cholesky 数值稳定
    on_progress
        每次新评估完调用：``on_progress(eval_index, x_best_so_far, f_best_so_far)``

    Returns
    -------
    ``(x_best, f_best)`` —— ``f_best`` 极端情况下可能是 ``+inf``（所有评估都失败）。
    """
    if rng is None:
        rng = np.random.default_rng(seed=12345)
    if length_scale is None:
        length_scale = max((b - a) / 5.0, 1e-9)
    if a >= b:
        raise ValueError(f"Invalid bracket: a={a}, b={b}")

    X_obs: List[float] = []
    y_obs: List[float] = []
    X_failed: List[float] = []
    eval_count = 0

    def _emit_progress() -> None:
        if on_progress is None:
            return
        try:
            if y_obs:
                i = int(np.argmin(y_obs))
                on_progress(eval_count, X_obs[i], y_obs[i])
            else:
                on_progress(eval_count, float("nan"), float("inf"))
        except Exception:
            pass

    def _record_observation(x_eval: float, y_eval: float) -> None:
        nonlocal eval_count
        if math.isfinite(y_eval):
            X_obs.append(float(x_eval))
            y_obs.append(float(y_eval))
        else:
            X_failed.append(float(x_eval))
        eval_count += 1
        _emit_progress()

    # 1) Warm start：不计入 eval_count（caller 已经跑过）
    if warm_start:
        for x, y in warm_start:
            try:
                x_f = float(x)
                y_f = float(y)
            except (TypeError, ValueError):
                continue
            if not (a <= x_f <= b):
                continue
            if math.isfinite(y_f):
                X_obs.append(x_f)
                y_obs.append(y_f)
            else:
                X_failed.append(x_f)

    # 2) 观测不足以训 GP 时，先做均匀初采样（真的调用 func）
    needed_initial = max(2, int(n_initial)) - len(X_obs)
    if needed_initial > 0 and eval_count < max_evaluations:
        seeds = np.linspace(a, b, needed_initial + 2)[1:-1]
        for cx in seeds:
            if eval_count >= max_evaluations:
                break
            cy = func(float(cx))
            _record_observation(float(cx), float(cy))

    # 3) 主 BO 循环
    while eval_count < max_evaluations:
        if y_obs:
            ymax = max(y_obs)
            ymin = min(y_obs)
            penalty = ymax + 10.0 * max(ymax - ymin, abs(ymax) + 1.0)
        else:
            penalty = 1.0

        X_train = np.array(X_obs + X_failed, dtype=float)
        y_train = np.array(y_obs + [penalty] * len(X_failed), dtype=float)

        if X_train.size < 1:
            next_x = float(rng.uniform(a, b))
        else:
            grid = np.linspace(a, b, int(candidate_grid))
            mu, sigma = _gp_posterior(
                X_train, y_train, grid, length_scale, noise_level
            )
            y_best = min(y_obs) if y_obs else penalty
            ei = _expected_improvement(mu, sigma, y_best, xi=exploration_xi)

            # 抑制贴近已采样点的 EI（半径取 length_scale 的一小部分）
            close_r = max(length_scale * 0.05, (b - a) * 1e-4)
            for x_sampled in X_train:
                ei[np.abs(grid - x_sampled) < close_r] = -np.inf

            if not np.any(np.isfinite(ei)) or np.max(ei) <= -np.inf:
                # 全被压成 -inf：兜底随机
                next_x = float(rng.uniform(a, b))
            else:
                next_x = float(grid[int(np.argmax(ei))])

        next_x = max(a, min(b, next_x))
        next_y = func(float(next_x))
        _record_observation(float(next_x), float(next_y))

    # 4) 取全局最优
    if y_obs:
        i_best = int(np.argmin(y_obs))
        return X_obs[i_best], y_obs[i_best]
    return float("nan"), float("inf")
