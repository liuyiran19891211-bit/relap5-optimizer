"""
Time-integral objectives for stripf_plotrec_rows.csv style outputs.

Used by auto_optimize when minimizing ∫ f(p(t) - p_ref) dt with trapezoidal rule.
"""

from __future__ import annotations

import csv
import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

DeviationMode = Literal["absolute", "signed", "squared"]


def select_best_value_column(
    csv_path: str,
    time_col: int,
    reference_value: float,
    configured_col: Optional[int] = None,
) -> Tuple[int, Optional[str]]:
    """
    为一条 plotrec CSV 选出"最该被作为优化变量"的列。

    判定顺序：
      1. 若 ``configured_col`` 落在 ``[0, ncols)`` 内：直接采用，返回 (col, None)。
      2. 否则按 **log10(|首行值|) 与 log10(|reference_value|) 之差最小** 选列
         （跳过 time_col、跳过非数值列）。返回 (col, 详细 warning 字符串)。
      3. 极端兜底：CSV 缺少有效数值列时，回退到最后一列。

    设计理由：在 plotrec CSV 里，**首行通常是 t=0 时的初值**，与 RELAP 输入直接相关；
    用 log10 距离衡量"量级匹配"对工程量比线性距离稳健（压力 ~1e7、流量 ~1e4、
    温度 ~5e2 三个数量级一目了然，绝不会把 1.5e7 配成 5e2）。
    """
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            first_row = next(reader, None)
    except OSError:
        return ((configured_col if configured_col is not None else 0),
                f"无法读取 CSV：{csv_path}")

    if not header:
        return ((configured_col if configured_col is not None else 0),
                f"CSV 表头为空：{csv_path}")

    ncols = len(header)
    cfg = configured_col if configured_col is not None else -1

    if 0 <= cfg < ncols:
        return (cfg, None)

    if not first_row:
        fallback = ncols - 1 if ncols > 0 else 0
        return (fallback,
                f"CSV 无数据行；configured_col={cfg} 越界（ncols={ncols}）；"
                f"回退到最后一列 #{fallback}（'{header[fallback]}'）")

    candidates: List[Tuple[int, float]] = []
    for i in range(ncols):
        if i == int(time_col):
            continue
        try:
            v = float(first_row[i])
        except (ValueError, IndexError):
            continue
        candidates.append((i, v))

    if not candidates:
        fallback = ncols - 1
        return (fallback,
                f"CSV 无可解析的数值列；configured_col={cfg} 越界（ncols={ncols}）；"
                f"回退到最后一列 #{fallback}（'{header[fallback]}'）")

    if not (math.isfinite(reference_value) and reference_value != 0):
        chosen = candidates[0][0]
        return (chosen,
                f"reference_value 不可用（{reference_value!r}）；"
                f"configured_col={cfg} 越界（ncols={ncols}）；"
                f"回退到首个数值列 #{chosen}（'{header[chosen]}'）")

    log_ref = math.log10(abs(reference_value))

    def score(item: Tuple[int, float]) -> float:
        _, v = item
        if v == 0 or not math.isfinite(v):
            return float("inf")
        return abs(math.log10(abs(v)) - log_ref)

    best_idx, best_val = min(candidates, key=score)
    return (best_idx,
            f"value_column_index={cfg} 越界（ncols={ncols}）；"
            f"按 reference_value={reference_value:g} 自动选中第 {best_idx} 列 "
            f"'{header[best_idx]}'（首行值 {best_val:g}）")


def read_csv_time_series(
    csv_path: str, time_col: int, value_col: int
) -> Tuple[List[float], List[float]]:
    """Read two columns from a RELAP plotrec CSV (header row + numeric rows)."""
    times: List[float] = []
    values: List[float] = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"Empty CSV: {csv_path}")
        ncols = len(header)
        if time_col < 0 or time_col >= ncols:
            raise IndexError(f"time_column_index {time_col} out of range (ncols={ncols})")
        if value_col < 0 or value_col >= ncols:
            raise IndexError(f"value_column_index {value_col} out of range (ncols={ncols})")

        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            if len(row) < max(time_col, value_col) + 1:
                continue
            try:
                t = float(row[time_col])
                v = float(row[value_col])
            except ValueError:
                continue
            times.append(t)
            values.append(v)

    if len(times) < 2:
        raise RuntimeError(
            f"Need at least two valid rows for integration; got {len(times)} from {csv_path}"
        )
    return times, values


def trapezoid_integral(x: List[float], y: List[float]) -> float:
    """∫ y dx using trapezoidal rule (x must be strictly increasing)."""
    if len(x) != len(y):
        raise ValueError("x and y length mismatch")
    if len(x) < 2:
        return 0.0
    total = 0.0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        if dx <= 0:
            raise ValueError(f"Time not strictly increasing at index {i}: {x[i]} -> {x[i + 1]}")
        total += 0.5 * (y[i] + y[i + 1]) * dx
    return total


def deviation_series(
    values: List[float], reference: float, mode: DeviationMode
) -> List[float]:
    if mode == "signed":
        return [v - reference for v in values]
    if mode == "absolute":
        return [abs(v - reference) for v in values]
    if mode == "squared":
        return [(v - reference) ** 2 for v in values]
    raise ValueError(f"Unknown deviation mode: {mode}")


def compute_time_integral_objective(
    csv_path: str,
    time_column_index: int,
    value_column_index: int,
    reference_value: float,
    deviation: DeviationMode = "absolute",
) -> float:
    """
    Scalar objective: trapezoidal ∫ g(p(t) - p_ref) dt where g depends on deviation mode.

    - absolute: ∫ |p - p_ref| dt  (default; minimizes accumulated pressure error magnitude)
    - signed:   ∫ (p - p_ref) dt  (literal difference; can cancel or blow up — use with care)
    - squared:  ∫ (p - p_ref)^2 dt
    """
    times, values = read_csv_time_series(
        csv_path, time_column_index, value_column_index
    )
    integrand = deviation_series(values, reference_value, deviation)
    return trapezoid_integral(times, integrand)


def compute_weighted_objective(
    csv_path: str,
    time_column_index: int,
    objectives: Sequence[Dict[str, Any]],
) -> Tuple[float, List[float]]:
    """
    多目标加权和标量化。

    每个 ``objectives`` 元素必须包含：
      - ``value_column_index`` : int
      - ``reference_value``    : float
      - ``deviation``          : ``"absolute"`` | ``"signed"`` | ``"squared"``（缺省 ``absolute``）
      - ``weight``             : float（缺省 1.0）
      - ``scale``              : float（缺省 1.0；用于把不同量级目标归一化到同等权重）
      - ``name`` (可选)        : str，仅用于诊断/UI

    Returns
    -------
    ``(aggregate, integrals)``
        - ``aggregate``  = ``Σ weight_i * (integral_i / scale_i)``
        - ``integrals``  = 每个目标在加权前的原始积分（保留全部信息供 UI 拆解）

    所有积分共用同一个 ``time_column_index``——同一次 RELAP 求解里时间轴只有一条。
    """
    if not objectives:
        raise ValueError("objectives 不能为空；至少需要 1 个目标")

    integrals: List[float] = []
    aggregate = 0.0
    for cfg in objectives:
        val_col = int(cfg["value_column_index"])
        ref = float(cfg["reference_value"])
        deviation = str(cfg.get("deviation", "absolute"))
        if deviation not in ("absolute", "signed", "squared"):
            raise ValueError(
                f"objectives[i].deviation 必须是 absolute|signed|squared，"
                f"得到 {deviation!r}"
            )
        integ = compute_time_integral_objective(
            csv_path,
            time_column_index,
            val_col,
            ref,
            deviation=deviation,  # type: ignore[arg-type]
        )
        integrals.append(integ)

        weight = float(cfg.get("weight", 1.0))
        scale = float(cfg.get("scale", 1.0))
        if scale == 0.0:
            raise ValueError(
                f"objectives[i].scale 不能为 0（'{cfg.get('name', '?')}'）"
            )
        aggregate += weight * (integ / scale)

    return aggregate, integrals


def golden_section_minimize(
    func,
    a: float,
    b: float,
    tol: float = 1e-4,
    max_evaluations: int = 40,
) -> Tuple[float, float]:
    """Minimize func(x) on [a, b] using golden-section search. Returns (x_best, f_best)."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio

    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)) or b <= a:
        raise ValueError(f"Invalid bracket: a={a}, b={b}")

    def feval(x: float) -> float:
        return float(func(x))

    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi
    f1 = feval(x1)
    f2 = feval(x2)

    eval_count = 2
    while eval_count < max_evaluations and (b - a) > tol * (1.0 + abs(a) + abs(b)):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - (b - a) / phi
            f1 = feval(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + (b - a) / phi
            f2 = feval(x2)
        eval_count += 1

    if f1 < f2:
        x_best, f_best = x1, f1
    else:
        x_best, f_best = x2, f2
    return x_best, f_best
