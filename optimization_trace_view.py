"""
RELAP5 参数优化 — 优化轨迹可视化（Plotly）。

职责（独立于 Streamlit / RELAP，便于单元测试）：
1. 每次评估的 p(t) 时间序列对比 + p_ref 参考线 + 偏差 |p−p_ref| 填充区
2. 参数轴上的搜索过程：评估点散点 + 评估顺序色阶 + 当前最优 + 初始区间带
3. 收敛曲线：本次评估目标 + 历史最优（cumulative best）
4. 多参数（N≥2）：任选两参数轴的二维目标热图（IDW 插值 + 评估点叠加）

并提供：
- runs/<run_id>/ 目录的创建（每次启动一个快照目录）
- 在 on_iteration 时刻把当前 CSV 拷贝为 eval_<n>.csv 的小工具
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go


# ---- 快照工具 ----------------------------------------------------------------

def make_run_snapshot_dir(base_dir: str, env_label: str) -> str:
    """在 base_dir/runs/ 下创建 <env>_<时间戳> 子目录，返回绝对路径。"""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_env = "".join(c for c in str(env_label) if c.isalnum() or c in ("_", "-")) or "run"
    out = os.path.join(base_dir, "runs", f"{safe_env}_{stamp}")
    os.makedirs(out, exist_ok=True)
    return out


def snapshot_eval_csv(
    src_csv_abs: str, snapshot_dir: str, evaluation_number: int
) -> Optional[str]:
    """把 src_csv 拷贝成 <snapshot_dir>/eval_<n>.csv。失败/缺源时返回 None。"""
    if not src_csv_abs or not os.path.isfile(src_csv_abs):
        return None
    try:
        n = int(evaluation_number)
    except (TypeError, ValueError):
        n = 0
    dst = os.path.join(snapshot_dir, f"eval_{n:03d}.csv")
    try:
        shutil.copy2(src_csv_abs, dst)
    except OSError:
        return None
    return dst


# ---- 历史记录字段适配（target_band / minimize_time_integral 兼容） -------------

def eval_id_of(item: Dict[str, Any]) -> int:
    if "evaluation" in item:
        try:
            return int(item["evaluation"])
        except (TypeError, ValueError):
            return 0
    if "iteration" in item:
        try:
            return int(item["iteration"])
        except (TypeError, ValueError):
            return 0
    return 0


def objective_of(item: Dict[str, Any]) -> float:
    """
    minimize_time_integral 模式：直接返回 objective_value。
    target_band 模式：以 |error| 作为可比较的目标值（数值越小越接近达标）。
    """
    if "objective_value" in item:
        try:
            return float(item["objective_value"])
        except (TypeError, ValueError):
            return float("nan")
    if "error" in item:
        try:
            return abs(float(item["error"]))
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


def format_parameter_vector(item: Dict[str, Any]) -> str:
    """把评估记录中的参数向量压成紧凑文本，供 UI 表格/状态栏展示。"""
    vals = item.get("parameter_values")
    if vals is None and "parameter_value" in item:
        vals = [item["parameter_value"]]
    if not isinstance(vals, (list, tuple)):
        return "—"
    out: List[str] = []
    for value in vals:
        try:
            out.append(f"{float(value):.6g}")
        except (TypeError, ValueError):
            out.append(str(value))
    return "[" + ", ".join(out) + "]" if out else "—"


def flatten_pareto_front(pareto_front: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 Pareto 记录展平成适合 ``st.dataframe`` 和 CSV 导出的行。"""
    def _sort_key(item: Dict[str, Any]) -> float:
        try:
            return float(item.get("objective_value", float("inf")))
        except (TypeError, ValueError):
            return float("inf")

    rows: List[Dict[str, Any]] = []
    for row in pareto_front or []:
        flat: Dict[str, Any] = {
            "evaluation": row.get("evaluation"),
            "objective_value": row.get("objective_value"),
        }
        for i, v in enumerate(row.get("parameter_values", []) or []):
            flat[f"param_{i + 1}"] = v
        for i, v in enumerate(row.get("objective_values", []) or []):
            flat[f"objective_{i + 1}"] = v
        rows.append(flat)
    rows.sort(key=_sort_key)
    return rows


# ---- CSV 读取 ----------------------------------------------------------------

def read_csv_two_columns(
    csv_path: str, time_col: int, value_col: int
) -> Tuple[List[float], List[float]]:
    """读取 plotrec CSV 的 (t, v)。出错或缺失返回 ([], [])。"""
    times: List[float] = []
    values: List[float] = []
    if not csv_path or not os.path.isfile(csv_path):
        return times, values
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return times, values
        ncols = len(header)
        if not (0 <= int(time_col) < ncols) or not (0 <= int(value_col) < ncols):
            return times, values
        for row in reader:
            if len(row) < ncols:
                continue
            try:
                t = float(row[int(time_col)])
                v = float(row[int(value_col)])
            except ValueError:
                continue
            times.append(t)
            values.append(v)
    return times, values


def abs_dev_integral(times: List[float], values: List[float], ref: float) -> float:
    """∫|v(t)−ref| dt（梯形法；时间须递增；非递增段跳过）。"""
    if len(times) != len(values) or len(times) < 2:
        return 0.0
    total = 0.0
    for i in range(len(times) - 1):
        dx = times[i + 1] - times[i]
        if dx <= 0:
            continue
        a = abs(values[i] - ref)
        b = abs(values[i + 1] - ref)
        total += 0.5 * (a + b) * dx
    return total


# ---- 图构建器 ----------------------------------------------------------------

_REF_LINE_COLOR = "#3a7bd5"
_FOCUS_FILL = "rgba(255, 99, 71, 0.18)"
_BRACKET_FILL = "rgba(58, 123, 213, 0.08)"
_BEST_COLOR = "#e74c3c"


def build_trace_compare_figure(
    history: List[Dict[str, Any]],
    snapshots: Dict[int, str],
    time_col: int,
    value_col: int,
    reference_value: float,
    selected_eval_ids: Optional[List[int]] = None,
    focus_eval_id: Optional[int] = None,
    title: str = "p(t) 对比 · 参考线 · 偏差填充",
) -> go.Figure:
    """
    给定多次评估的 CSV 快照路径，绘制：
    - 每条选中的 p(t) 曲线（focus 的高亮加粗）
    - 横向 p_ref 参考虚线
    - focus 曲线和 p_ref 之间填充区（直观看「∫|p−p_ref|dt」面积）
    """
    fig = go.Figure()

    if not history:
        fig.update_layout(
            title=title + "（暂无数据）",
            xaxis_title="时间 t",
            yaxis_title="输出参数 p(t)",
            height=420,
        )
        return fig

    if selected_eval_ids is None:
        all_ids = sorted({eval_id_of(item) for item in history})
        if len(all_ids) <= 3:
            selected_eval_ids = list(all_ids)
        else:
            selected_eval_ids = [all_ids[0], all_ids[len(all_ids) // 2], all_ids[-1]]

    x_min: float = math.inf
    x_max: float = -math.inf
    drew_any_curve = False

    for eval_id in selected_eval_ids:
        eid = int(eval_id)
        csv_path = snapshots.get(eid)
        if not csv_path:
            continue
        times, values = read_csv_two_columns(csv_path, time_col, value_col)
        if not times:
            continue
        x_min = min(x_min, times[0])
        x_max = max(x_max, times[-1])
        drew_any_curve = True

        item = next((h for h in history if eval_id_of(h) == eid), None)
        param_v = float(item["parameter_value"]) if item and "parameter_value" in item else float("nan")
        obj_v = objective_of(item) if item else float("nan")
        legend = f"#{eid} · param={param_v:.4g}"
        if math.isfinite(obj_v):
            legend += f" · obj={obj_v:.4g}"

        is_focus = focus_eval_id is not None and int(focus_eval_id) == eid
        fig.add_trace(
            go.Scatter(
                x=times, y=values,
                mode="lines",
                name=legend,
                line=dict(width=3 if is_focus else 1.6),
                opacity=1.0 if is_focus else 0.85,
                hovertemplate="t=%{x:.6g}<br>p=%{y:.6g}<extra></extra>",
            )
        )

        if is_focus:
            fig.add_trace(
                go.Scatter(
                    x=times, y=[float(reference_value)] * len(times),
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=times, y=values,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    fill="tonexty",
                    fillcolor=_FOCUS_FILL,
                    name=f"|p−p_ref| 区域 #{eid}",
                    hoverinfo="skip",
                )
            )
            integral = abs_dev_integral(times, values, float(reference_value))
            fig.add_annotation(
                x=times[-1], y=values[-1],
                text=f"∫|Δ|dt ≈ {integral:.4g}",
                showarrow=True, arrowhead=2, ax=-40, ay=-30,
                font=dict(size=12, color="#b04a3b"),
            )

    if drew_any_curve and math.isfinite(x_min) and math.isfinite(x_max):
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max], y=[float(reference_value)] * 2,
                mode="lines",
                name=f"p_ref = {reference_value:.4g}",
                line=dict(color=_REF_LINE_COLOR, dash="dash", width=2),
                hovertemplate=f"p_ref = {float(reference_value):.6g}<extra></extra>",
            )
        )
    elif not drew_any_curve:
        fig.add_annotation(
            text="尚无可读取的评估快照（CSV）",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#888"),
        )

    fig.update_layout(
        title=title,
        xaxis_title="时间 t",
        yaxis_title="输出参数 p(t)",
        height=460,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=60, r=20, t=50, b=70),
    )
    return fig


def build_per_parameter_search_figures(
    history: List[Dict[str, Any]],
    parameter_names: List[str],
    parameter_bounds: List[Tuple[float, float]],
) -> List[go.Figure]:
    """
    多参数情形：为每个参数 i 单独生成一张 ``(x_i, f)`` 散点图，
    可视化 BO/golden-section 在该维度上的探索轨迹与历史最优。

    每张图复用 :func:`build_search_axis_figure` 的样式（颜色按评估号、最优★）。
    history 里取 ``parameter_values[i]`` 作 x 轴；若不存在则退化到 ``parameter_value``
    （旧的 1D 单参数记录）。
    """
    figs: List[go.Figure] = []
    n = len(parameter_names)
    if n == 0:
        return figs

    for i, (pname, bracket) in enumerate(zip(parameter_names, parameter_bounds)):
        # 把 history 投影到第 i 维：构造一个临时记录列表，保留 evaluation / objective_value
        proj: List[Dict[str, Any]] = []
        for item in history:
            obj = objective_of(item)
            if not math.isfinite(obj):
                continue
            pvals = item.get("parameter_values")
            if isinstance(pvals, (list, tuple)) and len(pvals) > i:
                xi = float(pvals[i])
            elif i == 0 and "parameter_value" in item:
                xi = float(item["parameter_value"])
            else:
                continue
            proj.append({
                "evaluation": eval_id_of(item),
                "iteration": eval_id_of(item),
                "parameter_value": xi,
                "objective_value": obj,
            })
        fig = build_search_axis_figure(
            proj, bracket=bracket,
            title=f"参数 {i+1}/{n} · {pname}：搜索轨迹",
        )
        figs.append(fig)
    return figs


def _extract_axis_pair_samples(
    history: List[Dict[str, Any]],
    axis_i: int,
    axis_j: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从历史中提取 ``(x_i, x_j, objective, eval_id)``，仅保留有限目标值且具备向量坐标。
    """
    xi_l: List[float] = []
    xj_l: List[float] = []
    f_l: List[float] = []
    id_l: List[int] = []
    for item in history:
        obj = objective_of(item)
        if not math.isfinite(obj):
            continue
        pvals = item.get("parameter_values")
        if not isinstance(pvals, (list, tuple)):
            continue
        if len(pvals) <= max(axis_i, axis_j):
            continue
        try:
            vi = float(pvals[axis_i])
            vj = float(pvals[axis_j])
        except (TypeError, ValueError):
            continue
        xi_l.append(vi)
        xj_l.append(vj)
        f_l.append(float(obj))
        id_l.append(eval_id_of(item))
    if not xi_l:
        return (
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0, dtype=int),
        )
    return (
        np.asarray(xi_l, dtype=float),
        np.asarray(xj_l, dtype=float),
        np.asarray(f_l, dtype=float),
        np.asarray(id_l, dtype=int),
    )


def _idw_grid_values(
    xs: np.ndarray,
    ys: np.ndarray,
    fs: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    power: float = 2.0,
    min_dist: float = 1e-9,
) -> np.ndarray:
    """
    对散点 ``(xs, ys) -> fs`` 做反距离权重插值，得到 ``len(y_grid) × len(x_grid)`` 的 ``Z``。
    ``Z[row, col]`` 对应 ``(x_grid[col], y_grid[row])``，与 Plotly ``Heatmap`` 约定一致。
    """
    ny = len(y_grid)
    nx = len(x_grid)
    z_out = np.full((ny, nx), np.nan, dtype=float)
    if len(xs) == 0:
        return z_out

    for iy in range(ny):
        yp = y_grid[iy]
        for ix in range(nx):
            xp = x_grid[ix]
            dx = xs - xp
            dy = ys - yp
            d = np.hypot(dx, dy)
            hit = d < min_dist
            if np.any(hit):
                z_out[iy, ix] = float(fs[np.argmax(hit)])
                continue
            w = 1.0 / np.power(d, power)
            z_out[iy, ix] = float(np.sum(w * fs) / np.sum(w))
    return z_out


def build_parameter_pair_objective_heatmap(
    history: List[Dict[str, Any]],
    parameter_names: List[str],
    parameter_bounds: List[Tuple[float, float]],
    axis_i: int,
    axis_j: int,
    grid_resolution: int = 40,
    colorscale: str = "Viridis",
    title: Optional[str] = None,
) -> go.Figure:
    """
    **N≥2 参数**：任选两个参数轴，将聚合目标值映射为色阶。

    - 背景：对已成功评估的 ``(x_i, x_j)`` 点做 IDW 插值，绘制二维 ``Heatmap``（稀疏样本下为近似「地貌」）。
    - 前景：实际评估点散点（评估编号标注），便于对照 BO 采样位置。

    说明：加权标量目标下的「Pareto 前沿」需固定其余维度才能讨论；本图用于直观观察
    在当前历史采样下、沿所选两维投影的目标分布形态。
    """
    fig = go.Figure()
    npar = len(parameter_names)
    if npar < 2:
        fig.update_layout(
            title="参数平面热图（至少需要 2 个参数）",
            height=420,
            annotations=[dict(
                text="当前为单参数模式，请使用「参数轴搜索」散点图。",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=13, color="#888"),
            )],
        )
        return fig

    if axis_i == axis_j:
        fig.update_layout(
            title="请选择两个不同的参数轴",
            height=420,
        )
        return fig

    if not (
        0 <= axis_i < npar
        and 0 <= axis_j < npar
        and axis_i < len(parameter_bounds)
        and axis_j < len(parameter_bounds)
    ):
        fig.update_layout(title="参数轴索引无效", height=420)
        return fig

    ai, bi = float(parameter_bounds[axis_i][0]), float(parameter_bounds[axis_i][1])
    aj, bj = float(parameter_bounds[axis_j][0]), float(parameter_bounds[axis_j][1])
    if not (math.isfinite(ai) and math.isfinite(bi) and bi > ai):
        ai, bi = 0.0, 1.0
    if not (math.isfinite(aj) and math.isfinite(bj) and bj > aj):
        aj, bj = 0.0, 1.0

    xs, ys, fs, eids = _extract_axis_pair_samples(history, axis_i, axis_j)
    name_i = str(parameter_names[axis_i])
    name_j = str(parameter_names[axis_j])

    if title is None:
        title = f"目标值热图 · {name_i} × {name_j}"

    if len(fs) == 0:
        fig.update_layout(
            title=title + "（暂无有效投影点）",
            xaxis_title=name_i,
            yaxis_title=name_j,
            height=460,
            annotations=[dict(
                text="尚无带 parameter_values 且目标有限的评估记录。",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=13, color="#888"),
            )],
        )
        return fig

    res = max(12, min(80, int(grid_resolution)))
    x_lin = np.linspace(ai, bi, res)
    y_lin = np.linspace(aj, bj, res)
    z_grid = _idw_grid_values(xs, ys, fs, x_lin, y_lin)

    z_min = float(np.nanmin(z_grid))
    z_max = float(np.nanmax(z_grid))
    if z_max <= z_min:
        z_max = z_min + 1e-12

    fig.add_trace(
        go.Heatmap(
            x=x_lin,
            y=y_lin,
            z=z_grid,
            zsmooth=False,
            colorscale=colorscale,
            zmin=z_min,
            zmax=z_max,
            opacity=0.92,
            hovertemplate=(
                f"{name_j}=%{{y:.6g}}<br>{name_i}=%{{x:.6g}}"
                "<br>插值目标≈%{z:.6g}<extra></extra>"
            ),
            colorbar=dict(title="目标值"),
            name="插值地貌",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=[str(int(e)) for e in eids],
            textposition="top center",
            customdata=np.stack([eids.astype(float), fs], axis=1),
            marker=dict(
                size=11,
                color=fs,
                colorscale=colorscale,
                cmin=z_min,
                cmax=z_max,
                line=dict(width=1.5, color="rgba(255,255,255,0.95)"),
                showscale=False,
            ),
            name="评估点",
            hovertemplate=(
                "评估 #%{customdata[0]:.0f}<br>"
                f"{name_i}=%{{x:.6g}}<br>{name_j}=%{{y:.6g}}"
                "<br>目标=%{customdata[1]:.6g}<extra></extra>",
            ),
        )
    )

    i_best = int(np.argmin(fs))
    fig.add_trace(
        go.Scatter(
            x=[float(xs[i_best])],
            y=[float(ys[i_best])],
            mode="markers",
            marker=dict(
                size=18,
                color=_BEST_COLOR,
                symbol="star",
                line=dict(width=1, color="#7a1d12"),
            ),
            name=f"当前最优 #{int(eids[i_best])}",
            hovertemplate=(
                f"BEST #{int(eids[i_best])}<br>{name_i}=%{{x:.6g}}<br>"
                f"{name_j}=%{{y:.6g}}<br>目标=%{{customdata[0]:.6g}}<extra></extra>"
            ),
            customdata=[[float(fs[i_best])]],
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=name_i,
        yaxis_title=name_j,
        height=480,
        margin=dict(l=60, r=40, t=55, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0),
    )
    return fig


def build_search_axis_figure(
    history: List[Dict[str, Any]],
    bracket: Tuple[float, float],
    title: str = "参数轴上的搜索过程",
) -> go.Figure:
    """
    在 (参数 x, 目标 f) 平面上散点 + 折线，展现搜索轨迹。
    - 颜色按评估号渐变（Viridis）
    - 标记尺寸随评估号增长（最新点最显眼）
    - 当前最优用红色五角星标出
    - 初始区间 [a, b] 用淡蓝色背景带
    """
    fig = go.Figure()
    if not history:
        fig.update_layout(
            title=title + "（暂无评估点）",
            xaxis_title="参数 x",
            yaxis_title="目标 f(x)",
            height=380,
        )
        return fig

    pairs: List[Tuple[int, float, float]] = []
    for item in history:
        try:
            x = float(item.get("parameter_value"))
        except (TypeError, ValueError):
            continue
        y = objective_of(item)
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((eval_id_of(item), x, y))

    if not pairs:
        fig.update_layout(title=title + "（无有效目标值）", height=380)
        return fig

    ids = [p[0] for p in pairs]
    xs = [p[1] for p in pairs]
    ys = [p[2] for p in pairs]

    a, b = float(bracket[0]), float(bracket[1])
    if math.isfinite(a) and math.isfinite(b) and b > a:
        fig.add_vrect(
            x0=a, x1=b,
            fillcolor=_BRACKET_FILL,
            line_width=0,
            annotation_text=f"初始区间 [{a:g}, {b:g}]",
            annotation_position="top left",
        )

    base_size = 10
    sizes = [base_size + 1.5 * i for i in range(len(xs))]

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="markers+lines+text",
            text=[str(n) for n in ids],
            textposition="top center",
            marker=dict(
                size=sizes,
                color=ids,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="评估号"),
                line=dict(width=1, color="rgba(50,50,50,0.6)"),
            ),
            line=dict(color="rgba(120,120,120,0.45)", width=1),
            name="评估轨迹",
            hovertemplate="#%{text} · x=%{x:.6g} · f=%{y:.6g}<extra></extra>",
        )
    )

    i_best = min(range(len(ys)), key=lambda i: ys[i])
    fig.add_trace(
        go.Scatter(
            x=[xs[i_best]], y=[ys[i_best]],
            mode="markers",
            marker=dict(size=20, color=_BEST_COLOR, symbol="star",
                        line=dict(width=1, color="#7a1d12")),
            name=f"当前最优 #{ids[i_best]}",
            hovertemplate=(
                f"BEST #{ids[i_best]} · x=%{{x:.6g}} · f=%{{y:.6g}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="参数 x",
        yaxis_title="目标 f(x)",
        height=420,
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig


def build_live_pt_figure(
    csv_path: str,
    time_col: int,
    value_col: int,
    reference_value: float,
    title: str = "实时计算预览 · p(t)",
    last_updated_at: Optional[float] = None,
) -> go.Figure:
    """
    单条 ``p(t)`` 曲线 + ``p_ref`` 参考虚线 + ``|p−p_ref|`` 填充区，
    标注当前评估的瞬时积分与最近更新时间。CSV 缺失时返回带占位文字的空图。
    """
    fig = go.Figure()
    times, values = read_csv_two_columns(csv_path, time_col, value_col)

    if not times:
        fig.update_layout(
            title=title + "（暂无数据）",
            xaxis_title="时间 t",
            yaxis_title="输出参数 p(t)",
            height=380,
            margin=dict(l=60, r=20, t=50, b=60),
            annotations=[dict(
                text="尚未生成实时 CSV，等待 RELAP 写出 rstplt …",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=13, color="#888"),
            )],
        )
        return fig

    ref = float(reference_value)
    integral = abs_dev_integral(times, values, ref)

    fig.add_trace(go.Scatter(
        x=[times[0], times[-1]], y=[ref, ref],
        mode="lines",
        name=f"p_ref = {ref:.4g}",
        line=dict(color=_REF_LINE_COLOR, dash="dash", width=2),
        hovertemplate=f"p_ref = {ref:.6g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=[ref] * len(times),
        mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=values,
        mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=0),
        fill="tonexty",
        fillcolor=_FOCUS_FILL,
        name="|p−p_ref| 填充",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=values,
        mode="lines",
        name="p(t)",
        line=dict(color="#2c3e50", width=2.4),
        hovertemplate="t=%{x:.6g}<br>p=%{y:.6g}<extra></extra>",
    ))

    fig.add_annotation(
        x=times[-1], y=values[-1],
        text=f"∫|Δ|dt ≈ {integral:.4g} · n={len(times)}",
        showarrow=True, arrowhead=2, ax=-40, ay=-30,
        font=dict(size=12, color="#b04a3b"),
    )

    ts_label = ""
    if last_updated_at is not None:
        try:
            ts_label = " · 最近更新 " + time.strftime(
                "%H:%M:%S", time.localtime(float(last_updated_at))
            )
        except (TypeError, ValueError):
            ts_label = ""

    fig.update_layout(
        title=title + ts_label,
        xaxis_title="时间 t",
        yaxis_title="输出参数 p(t)",
        height=380,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=60, r=20, t=50, b=70),
    )
    return fig


def extract_integrals(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    抽取所有「成功的积分模式评估」，按发生顺序返回。

    每条记录包含：
      - ``eval_id``    : 评估号
      - ``parameter``  : 实际成功的参数值
      - ``integral``   : ``∫|p(t)−p_ref|dt``（即 ``objective_value``，已确认 finite）
      - ``is_best``    : 截止当前是否刷新了历史最优（cum-best）
      - ``retries``    : 该评估内 RELAP 重试次数（0=首跑成功）

    过滤规则：跳过 ``relap_failed`` / ``post_run_error`` / 缺失 ``objective_value`` /
    ``inf`` / ``NaN`` 的记录。仅适用于 ``minimize_time_integral`` 模式
    （其他模式不会写 ``objective_value``，自然返回空列表）。
    """
    out: List[Dict[str, Any]] = []
    best_so_far = math.inf
    for item in history:
        if item.get("relap_failed") or item.get("post_run_error"):
            continue
        if "objective_value" not in item:
            continue
        try:
            obj = float(item["objective_value"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(obj):
            continue
        is_best = obj <= best_so_far
        if is_best:
            best_so_far = obj
        try:
            param = float(item.get("parameter_value"))
        except (TypeError, ValueError):
            param = float("nan")
        out.append({
            "eval_id": eval_id_of(item),
            "parameter": param,
            "integral": obj,
            "is_best": bool(is_best),
            "retries": int(item.get("retries", 0) or 0),
        })
    return out


_INTEGRAL_BEST_COLOR = "#27ae60"
_INTEGRAL_NORMAL_COLOR = "#7d8a99"


def build_integral_bar_figure(
    history: List[Dict[str, Any]],
    metric_label: str = "∫|p(t)−p_ref|dt",
    title: Optional[str] = None,
) -> go.Figure:
    """
    每次成功评估的"偏差时间积分"柱状图。

    - 高度 = 当次评估的 ``integral``
    - 颜色 = 绿（刷新了历史最优）/ 灰（未刷新）
    - 柱顶标注实际数值
    """
    rows = extract_integrals(history)
    fig = go.Figure()
    if title is None:
        title = f"成功评估 · {metric_label} · 绿色柱 = 当时刷新历史最优"

    if not rows:
        fig.update_layout(
            title=title + "（暂无成功评估）",
            xaxis_title="评估编号",
            yaxis_title=metric_label,
            height=320,
            margin=dict(l=60, r=20, t=50, b=50),
            annotations=[dict(
                text="尚未捕获成功的积分评估。完成首次成功评估后这里会出现柱状图。",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=12, color="#888"),
            )],
        )
        return fig

    eids = [r["eval_id"] for r in rows]
    integrals = [r["integral"] for r in rows]
    colors = [
        _INTEGRAL_BEST_COLOR if r["is_best"] else _INTEGRAL_NORMAL_COLOR
        for r in rows
    ]

    fig.add_trace(go.Bar(
        x=eids,
        y=integrals,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.4g}" for v in integrals],
        textposition="outside",
        hovertemplate=(
            "评估 #%{x}<br>"
            f"{metric_label} = %{{y:.6g}}<extra></extra>"
        ),
        name=metric_label,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="评估编号",
        yaxis_title=metric_label,
        xaxis=dict(type="category"),
        height=340,
        margin=dict(l=60, r=20, t=50, b=50),
        showlegend=False,
    )
    return fig


def compute_live_stats(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    从优化历史实时计算关键指标，供仪表盘顶部 metric 卡片在评估**进行中**就能展示。

    输出字段
    --------
    - ``total``                 : 已完成评估总数
    - ``failed``                : RELAP 失败 / 后处理失败的评估数
    - ``latest_obj``            : 最近一次评估的目标值（None=最近一次失败/不可用）
    - ``latest_param``          : 最近一次评估的实际参数值
    - ``latest_eval_id``        : 最近一次评估号
    - ``best_obj``              : 当前历史最优目标值（cum-best；跳过 inf/nan/失败）
    - ``best_param``            : 取到 best_obj 时的参数值
    - ``best_eval_id``          : 取到 best_obj 时的评估号
    - ``delta_vs_prev_best``    : ``latest_obj - prev_best``（excluding latest）；
                                  负数=本次评估打破了之前的纪录；正数=不如之前最优；
                                  0=与之前最优持平；None=无可比较的历史
    """
    out: Dict[str, Any] = {
        "total": len(history),
        "failed": 0,
        "latest_obj": None,
        "latest_param": None,
        "latest_eval_id": None,
        "best_obj": None,
        "best_param": None,
        "best_eval_id": None,
        "delta_vs_prev_best": None,
    }
    if not history:
        return out

    valid: List[Tuple[int, float, Dict[str, Any]]] = []
    for i, item in enumerate(history):
        if item.get("relap_failed") or item.get("post_run_error"):
            out["failed"] += 1
        obj = objective_of(item)
        if math.isfinite(obj):
            valid.append((i, obj, item))

    last = history[-1]
    out["latest_param"] = last.get("parameter_value")
    out["latest_eval_id"] = eval_id_of(last)
    last_obj = objective_of(last)
    if math.isfinite(last_obj):
        out["latest_obj"] = last_obj

    if valid:
        best_idx, best_obj, best_item = min(valid, key=lambda t: t[1])
        out["best_obj"] = best_obj
        out["best_param"] = best_item.get("parameter_value")
        out["best_eval_id"] = eval_id_of(best_item)

        if out["latest_obj"] is not None:
            prev = [o for (i, o, _) in valid if i != len(history) - 1]
            if prev:
                prev_best = min(prev)
                out["delta_vs_prev_best"] = out["latest_obj"] - prev_best

    return out


def build_convergence_figure(
    history: List[Dict[str, Any]],
    log_y: bool = False,
    title: str = "目标值收敛",
) -> go.Figure:
    """单次评估目标 + 历史最优（cumulative best）。"""
    fig = go.Figure()
    if not history:
        fig.update_layout(title=title + "（暂无评估）", height=300)
        return fig

    ids = [eval_id_of(h) for h in history]
    fs = [objective_of(h) for h in history]

    cum_best: List[float] = []
    cur = math.inf
    for v in fs:
        if math.isfinite(v) and v < cur:
            cur = v
        cum_best.append(cur if math.isfinite(cur) else float("nan"))

    fig.add_trace(go.Scatter(
        x=ids, y=fs, mode="lines+markers",
        name="本次评估目标",
        line=dict(color="#888", width=1),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=ids, y=cum_best, mode="lines",
        name="历史最优 (cum-best)",
        line=dict(color="#27ae60", width=3),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="评估编号",
        yaxis_title="目标值",
        yaxis_type="log" if log_y else "linear",
        height=320,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig
