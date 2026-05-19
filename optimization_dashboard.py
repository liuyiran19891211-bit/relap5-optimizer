"""
RELAP5 参数优化 — 可视化交互控制台（Streamlit）。

运行（项目根目录）：
  pip install streamlit
  streamlit run optimization_dashboard.py
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from auto_optimize import (
    BASE_DIR,
    format_value,
    normalize_objectives_config,
    normalize_parameters_config,
    parse_stripf_records,
    run_optimization,
    validate_integral_optimization_config,
)
from live_preview import CSV_WRITE_LOCK, start_live_preview_loop
from optimization_trace_view import (
    build_convergence_figure,
    build_integral_bar_figure,
    build_live_pt_figure,
    build_parameter_pair_objective_heatmap,
    build_per_parameter_search_figures,
    build_search_axis_figure,
    build_trace_compare_figure,
    compute_live_stats,
    eval_id_of,
    extract_integrals,
    flatten_pareto_front,
    format_parameter_vector,
    make_run_snapshot_dir,
    snapshot_eval_csv,
)

LIVE_PREVIEW_CSV_REL = "plot/stripf_plotrec_row.csv"
LIVE_PREVIEW_INTERVAL_SEC = 20.0

CONFIG_FILES = {
    "dev": os.path.join(BASE_DIR, "config_dev.json"),
    "prod": os.path.join(BASE_DIR, "config_prod.json"),
}


def load_config_file(effective_env: str) -> Dict[str, Any]:
    path = CONFIG_FILES["prod" if effective_env == "prod" else "dev"]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config_file(effective_env: str, config: Dict[str, Any]) -> None:
    path = CONFIG_FILES["prod" if effective_env == "prod" else "dev"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


DEVIATION_LABELS = {
    "absolute": "绝对误差 ∫|p−p_ref| dt（推荐）",
    "signed": "有符号 ∫(p−p_ref) dt",
    "squared": "平方误差 ∫(p−p_ref)² dt",
}


def _default_integral_objective() -> Dict[str, Any]:
    return {
        "stripf_relative_path": "stripf",
        # 与「实时预览」共用同一份 CSV：优化器写它、实时预览也写它（互斥锁串行）。
        "csv_relative_path": LIVE_PREVIEW_CSV_REL,
        "time_column_index": 0,
        "value_column_index": 16,
        "reference_value": 15.5e6,
        "deviation": "absolute",
    }


def _default_integral_optimizer() -> Dict[str, Any]:
    return {
        "convergence_tolerance": 0.5,
        "max_function_evaluations": 25,
        "max_relap_retries": 10,
        "optimizer": "genetic",
        "exploration_xi": 0.01,
        "random_seed": 12345,
        "ga_crossover_probability": 0.9,
        "ga_mutation_scale": 0.12,
    }


OPTIMIZER_LABELS = {
    "golden_section": "黄金分割搜索（确定性，逐步收缩区间）",
    "bayesian": "贝叶斯优化 GP+EI（基于历史采样的 N-D 加速）",
    "genetic": "遗传算法（Pareto / NSGA-II，多目标多参数）",
}


def _format_initial_value_for_editor(value: Any) -> str:
    try:
        return format_value(float(value))
    except (TypeError, ValueError):
        return str(value)


def _format_param_rows_for_editor(rows: Any) -> List[Dict[str, Any]]:
    iterable = rows if isinstance(rows, list) else (
        rows.to_dict("records") if hasattr(rows, "to_dict") else []
    )
    formatted: List[Dict[str, Any]] = []
    for row in iterable:
        item = dict(row)
        if "initial_value" in item:
            item["initial_value"] = _format_initial_value_for_editor(
                item["initial_value"]
            )
        formatted.append(item)
    return formatted


def _params_to_rows(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """规范化为 ``st.data_editor`` 可消费的字典行列表。"""
    return [
        {
            "name": str(p.get("name", f"param_{i+1}")),
            "description": str(p.get("description", "")),
            "line_key": str(p.get("line_key", "")),
            "column_index": int(p.get("column_index", 0)),
            "initial_value": _format_initial_value_for_editor(
                p.get("initial_value", 0.0)
            ),
            "min_value": float(p.get("min_value", 0.0)),
            "max_value": float(p.get("max_value", 1.0)),
            "step": float(p.get("step", 1.0)),
        }
        for i, p in enumerate(params)
    ]


def _objectives_to_rows(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "name": str(o.get("name", f"obj_{i+1}")),
            "value_column_index": int(o.get("value_column_index", 0)),
            "reference_value": float(o.get("reference_value", 0.0)),
            "deviation": str(o.get("deviation", "absolute")),
            "weight": float(o.get("weight", 1.0)),
            "scale": float(o.get("scale", 1.0)),
        }
        for i, o in enumerate(objs)
    ]


def hydrate_from_config(cfg: Dict[str, Any]) -> None:
    st.session_state.objective = "minimize_time_integral"
    st.session_state.max_iterations = int(cfg.get("max_iterations", 5))

    # 多参数：用 normalize_parameters_config 兼容 legacy "parameter" / 新 "parameters"
    try:
        params = normalize_parameters_config(cfg)
    except KeyError:
        params = []
    st.session_state.params_rows = _params_to_rows(params)

    # legacy 单参数字段保留，只用于把老配置迁移成 parameters[] 的首行。
    p0 = params[0] if params else {}
    st.session_state.param_name = p0.get("name", "")
    st.session_state.param_description = p0.get("description", "")
    st.session_state.line_key = str(p0.get("line_key", ""))
    st.session_state.column_index = int(p0.get("column_index", 0))
    st.session_state.initial_value = float(p0.get("initial_value", 0.0))
    st.session_state.min_value = float(p0.get("min_value", 0.0))
    st.session_state.max_value = float(p0.get("max_value", 1.0))
    st.session_state.step_value = float(p0.get("step", 1.0))

    # 多目标：用 normalize_objectives_config 兼容 legacy / 新格式
    try:
        objs = normalize_objectives_config(cfg)
    except KeyError:
        objs = []
    st.session_state.objectives_rows = _objectives_to_rows(objs)

    # legacy IO 字段：用于"实时预览"那张图与摘要 metric（取首条）
    io = {**_default_integral_objective(), **cfg.get("integral_objective", {})}
    st.session_state.io_stripf_rel = str(io["stripf_relative_path"])
    st.session_state.io_csv_rel = str(io["csv_relative_path"])
    st.session_state.io_time_col = int(io["time_column_index"])
    o0 = objs[0] if objs else {}
    st.session_state.io_value_col = int(
        o0.get("value_column_index", io["value_column_index"])
    )
    st.session_state.io_reference = float(
        o0.get("reference_value", io["reference_value"])
    )
    st.session_state.io_deviation = str(
        o0.get("deviation", io.get("deviation", "absolute"))
    )

    iopt = {**_default_integral_optimizer(), **cfg.get("integral_optimizer", {})}
    st.session_state.io_conv_tol = float(iopt["convergence_tolerance"])
    st.session_state.io_max_eval = int(iopt["max_function_evaluations"])
    st.session_state.io_max_relap_retries = int(iopt.get("max_relap_retries", 10))
    st.session_state.io_optimizer = str(iopt.get("optimizer", "golden_section"))
    st.session_state.io_exploration_xi = float(iopt.get("exploration_xi", 0.01))
    st.session_state.io_random_seed = int(iopt.get("random_seed", 12345))
    st.session_state.io_ga_crossover_probability = float(
        iopt.get("ga_crossover_probability", 0.9)
    )
    st.session_state.io_ga_mutation_scale = float(
        iopt.get("ga_mutation_scale", 0.12)
    )


def _rows_to_parameters(rows: Any) -> List[Dict[str, Any]]:
    """把 ``st.data_editor`` 返回的可迭代对象（DataFrame 或 list[dict]）转成
    ``parameters`` 列表；剔除空行（line_key 为空）。"""
    out: List[Dict[str, Any]] = []
    iterable = rows if isinstance(rows, list) else (
        rows.to_dict("records") if hasattr(rows, "to_dict") else []
    )
    for i, r in enumerate(iterable):
        line_key = str(r.get("line_key", "")).strip()
        if not line_key:
            continue
        try:
            out.append({
                "name": str(r.get("name", f"param_{i+1}")) or f"param_{i+1}",
                "description": str(r.get("description", "")),
                "line_key": line_key,
                "column_index": int(r["column_index"]),
                "initial_value": float(r["initial_value"]),
                "min_value": float(r["min_value"]),
                "max_value": float(r["max_value"]),
                "step": float(r.get("step", 1.0)),
            })
        except (KeyError, TypeError, ValueError):
            continue  # 缺字段或类型错误的行跳过
    return out


def _rows_to_objectives(rows: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    iterable = rows if isinstance(rows, list) else (
        rows.to_dict("records") if hasattr(rows, "to_dict") else []
    )
    for i, r in enumerate(iterable):
        try:
            col = int(r["value_column_index"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            out.append({
                "name": str(r.get("name", f"obj_{i+1}")) or f"obj_{i+1}",
                "value_column_index": col,
                "reference_value": float(r["reference_value"]),
                "deviation": str(r.get("deviation", "absolute")),
                "weight": float(r.get("weight", 1.0)),
                "scale": float(r.get("scale", 1.0)),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out


def build_runtime_config(env_arg: str) -> Dict[str, Any]:
    """env_arg 传给 run_optimization；磁盘文件按 prod / dev 选择。"""
    effective = "prod" if env_arg == "prod" else "dev"
    cfg = load_config_file(effective)
    cfg["env"] = env_arg
    cfg["sleep_after_start"] = float(cfg.get("sleep_after_start", 0.0))
    cfg["objective"] = "minimize_time_integral"

    # 多参数：从 data_editor 行回收为 parameters 列表。
    params = _rows_to_parameters(st.session_state.get("params_rows", []))
    if not params:
        # 极端兜底：用从老配置迁移出的首行字段。
        params = [{
            "name": st.session_state.param_name,
            "description": st.session_state.param_description,
            "line_key": st.session_state.line_key.strip(),
            "column_index": int(st.session_state.column_index),
            "initial_value": float(st.session_state.initial_value),
            "min_value": float(st.session_state.min_value),
            "max_value": float(st.session_state.max_value),
            "step": float(st.session_state.step_value),
        }]
    cfg["parameters"] = params
    cfg.pop("parameter", None)
    cfg.pop("criterion", None)

    # 多目标：从 data_editor 行回收为 objectives 列表。
    objs = _rows_to_objectives(st.session_state.get("objectives_rows", []))
    if not objs:
        objs = [{
            "name": "obj_1",
            "value_column_index": int(st.session_state.io_value_col),
            "reference_value": float(st.session_state.io_reference),
            "deviation": str(st.session_state.io_deviation),
            "weight": 1.0,
            "scale": 1.0,
        }]
    cfg["objectives"] = objs

    first = objs[0]
    cfg["integral_objective"] = {
        "stripf_relative_path": st.session_state.io_stripf_rel.strip(),
        "csv_relative_path": st.session_state.io_csv_rel.strip(),
        "time_column_index": int(st.session_state.io_time_col),
        "value_column_index": int(first["value_column_index"]),
        "reference_value": float(first["reference_value"]),
        "deviation": str(first.get("deviation", "absolute")),
    }
    cfg["integral_optimizer"] = {
        "convergence_tolerance": float(st.session_state.io_conv_tol),
        "max_function_evaluations": int(st.session_state.io_max_eval),
        "max_relap_retries": int(st.session_state.io_max_relap_retries),
        "optimizer": str(st.session_state.io_optimizer),
        "exploration_xi": float(st.session_state.io_exploration_xi),
        "random_seed": int(st.session_state.io_random_seed),
        "ga_crossover_probability": float(
            st.session_state.io_ga_crossover_probability
        ),
        "ga_mutation_scale": float(st.session_state.io_ga_mutation_scale),
    }

    cfg["max_iterations"] = int(st.session_state.max_iterations)
    validate_integral_optimization_config(cfg)
    return cfg


def drain_worker_queues() -> None:
    log_q: "queue.Queue[str]" = st.session_state.log_q
    iter_q: "queue.Queue[Dict[str, Any]]" = st.session_state.iter_q
    result_q: "queue.Queue[Dict[str, Any]]" = st.session_state.result_q
    live_q: "queue.Queue[Dict[str, Any]]" = st.session_state.live_q

    while True:
        try:
            st.session_state.logs.append(log_q.get_nowait())
        except queue.Empty:
            break
    while True:
        try:
            item = iter_q.get_nowait()
        except queue.Empty:
            break
        st.session_state.history.append(item)
        snap = item.get("snapshot_csv")
        if snap:
            st.session_state.snapshot_paths[int(eval_id_of(item))] = snap
    while True:
        try:
            st.session_state.live_preview_meta = live_q.get_nowait()
        except queue.Empty:
            break
    while True:
        try:
            st.session_state.last_result = result_q.get_nowait()
            st.session_state.running = False
        except queue.Empty:
            break


def history_to_chart_rows(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in history:
        if "iteration" in item:
            out.append(
                {
                    "step": float(item["iteration"]),
                    "parameter_value": float(item["parameter_value"]),
                    "metric": float(item["criterion_value"]),
                    "abs_error": abs(float(item["error"])),
                    "kind": "容差迭代",
                }
            )
        else:
            fv = float(item["objective_value"])
            out.append(
                {
                    "step": float(item.get("evaluation", 0)),
                    "parameter_value": float(item["parameter_value"]),
                    "metric": fv,
                    "abs_error": abs(fv),
                    "kind": "积分评估",
                }
            )
    return out


def history_to_download_csv(history: List[Dict[str, Any]]) -> bytes:
    if not history:
        return b""
    rows = history_to_chart_rows(history)
    return rows_to_download_csv(rows)


def rows_to_download_csv(rows: List[Dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    sink = io.StringIO(newline="")
    writer = csv.DictWriter(sink, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return sink.getvalue().encode("utf-8-sig")


def _build_iteration_callback(
    iter_q: "queue.Queue[Dict[str, Any]]",
    csv_abs_path: str,
    snapshot_dir: str,
):
    """
    Wrap auto_optimize.on_iteration: snapshot the just-produced CSV, then enqueue.

    Runs in the worker thread (capture refs only — never touches st.session_state).
    """
    def _cb(item: Dict[str, Any]) -> None:
        try:
            n = item.get("evaluation", item.get("iteration", 0))
            # 与实时预览共用同一份 live.csv → 拷贝时也走 CSV_WRITE_LOCK，
            # 防止「正在被写入的 live.csv」被复制成残缺快照。
            with CSV_WRITE_LOCK:
                snap = snapshot_eval_csv(csv_abs_path, snapshot_dir, n)
            if snap:
                item = {**item, "snapshot_csv": snap}
        except Exception:
            pass
        iter_q.put(item)
    return _cb


def start_worker(env: str, config: Dict[str, Any]) -> None:
    st.session_state.stop_event = threading.Event()
    st.session_state.log_q = queue.Queue()
    st.session_state.iter_q = queue.Queue()
    st.session_state.result_q = queue.Queue()
    st.session_state.live_q = queue.Queue()
    st.session_state.logs = []
    st.session_state.history = []
    st.session_state.snapshot_paths = {}
    st.session_state.live_preview_meta = None
    st.session_state.last_result = None
    st.session_state.running = True

    snapshot_dir = make_run_snapshot_dir(BASE_DIR, env)
    st.session_state.snapshot_dir = snapshot_dir
    # 1D 摘要 bracket 取首参数；多参数下逐参数图自带各自的 bracket
    params_list = config.get("parameters") or [config.get("parameter", {})]
    first_p = params_list[0] if params_list else {}
    st.session_state.run_bracket = (
        float(first_p.get("min_value", 0.0)),
        float(first_p.get("max_value", 1.0)),
    )

    csv_rel = config.get("integral_objective", {}).get(
        "csv_relative_path", LIVE_PREVIEW_CSV_REL
    )
    csv_abs = os.path.join(BASE_DIR, str(csv_rel))

    iter_q_ref = st.session_state.iter_q
    log_q_ref = st.session_state.log_q
    result_q_ref = st.session_state.result_q
    live_q_ref = st.session_state.live_q
    stop_event_ref = st.session_state.stop_event
    live_preview_enabled = bool(st.session_state.get("live_preview_enabled", True))

    on_iteration_cb = _build_iteration_callback(iter_q_ref, csv_abs, snapshot_dir)

    def run_task() -> None:
        try:
            result = run_optimization(
                env=env,
                config_override=config,
                stop_requested=stop_event_ref.is_set,
                on_log=lambda m: log_q_ref.put(m),
                on_iteration=on_iteration_cb,
            )
        except Exception as exc:
            log_q_ref.put(f"[ERROR] {exc}")
            result = {
                "success": False,
                "stop_reason": "exception",
                "history": [],
                "final_parameter_value": None,
                "final_parameter_values": [],
                "final_objective_value": None,
                "final_objective_values": [],
                "pareto_front": [],
                "config": config,
            }
        result_q_ref.put(result)
        # 优化主流程结束后通知预览线程退出（避免 stop 按钮没按时拖尾运行）
        stop_event_ref.set()

    st.session_state.worker_thread = threading.Thread(target=run_task, daemon=True)
    st.session_state.worker_thread.start()

    if live_preview_enabled:
        st.session_state.live_preview_thread = start_live_preview_loop(
            stop_event=stop_event_ref,
            base_dir=BASE_DIR,
            output_csv_rel=LIVE_PREVIEW_CSV_REL,
            interval_sec=LIVE_PREVIEW_INTERVAL_SEC,
            on_log=lambda m: log_q_ref.put(m),
            on_update=lambda meta: live_q_ref.put(meta),
            grace_period_sec=8.0,
        )
    else:
        st.session_state.live_preview_thread = None


def parse_stripf_for_chart() -> List[Dict[str, float]]:
    stripf_path = os.path.join(BASE_DIR, "stripf")
    if not os.path.exists(stripf_path):
        return []
    records = parse_stripf_records(stripf_path)
    if not records:
        return []
    rows: List[Dict[str, float]] = []
    for rec in records:
        row: Dict[str, float] = {"col_0_time": float(rec[0])}
        for idx in range(1, len(rec)):
            row[f"col_{idx}"] = float(rec[idx])
        rows.append(row)
    return rows


def read_plot_csv_rows(rel_path: str) -> Tuple[Optional[List[str]], List[List[float]]]:
    fp = os.path.join(BASE_DIR, rel_path)
    if not os.path.isfile(fp):
        return None, []

    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if not headers:
            return None, []
        rows_text = list(reader)

    ncol = len(headers)
    data: List[List[float]] = []
    for parts in rows_text:
        if not parts or len(parts) < ncol:
            continue
        floats: List[float] = []
        ok = True
        for i in range(ncol):
            try:
                floats.append(float(parts[i]))
            except ValueError:
                ok = False
                break
        if ok:
            data.append(floats)
    return headers, data


def count_csv_data_rows(abs_path: str) -> int:
    if not os.path.isfile(abs_path):
        return 0
    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            return sum(1 for row in reader if row)
    except OSError:
        return 0


def csv_to_chart_dicts(headers: List[str], rows: List[List[float]]) -> List[Dict[str, Any]]:
    if not rows or not headers:
        return []

    hdr_safe: List[str] = []
    seen: Dict[str, int] = {}
    for i, h in enumerate(headers):
        base = str(h).strip() or f"col_{i}"
        n = seen.get(base, 0)
        seen[base] = n + 1
        hdr_safe.append(base if n == 0 else f"{base}__{n}")

    return [dict(zip(hdr_safe, r)) for r in rows]


def chart_rows_with_x(history_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{**r, "_x": float(r["step"])} for r in history_rows]


def render_csv_charts(csv_rel: str) -> None:
    hdrs, frows = read_plot_csv_rows(csv_rel)
    if not hdrs or not frows:
        st.warning(f"无法读取或未解析数值行：**`{csv_rel}`**")
        return

    chart_data = csv_to_chart_dicts(hdrs, frows)
    colnames_order = list(chart_data[0].keys())

    ti = int(st.session_state.io_time_col)
    vi = int(st.session_state.io_value_col)

    tc = colnames_order[ti] if 0 <= ti < len(colnames_order) else colnames_order[0]
    vc = colnames_order[vi] if 0 <= vi < len(colnames_order) else colnames_order[-1]

    ti_idx = colnames_order.index(tc)

    left, mid = st.columns((1.0, 1.85))
    with left:
        c_time_name = st.selectbox(
            "横轴 · 时间列",
            colnames_order,
            index=ti_idx,
            key="csv_time_col_pick",
        )
    with mid:
        st.markdown(f"_共 **{len(chart_data)}** 点 · **{len(colnames_order)}** 列_")

    other = [n for n in colnames_order if n != c_time_name]
    vc_idx = colnames_order.index(vc) if vc in colnames_order else 0

    picks = st.multiselect(
        "纵轴 · 可多选绘图列（至少一列）",
        other,
        default=[vc if vc != c_time_name else other[max(0, min(vc_idx, len(other) - 1))]],
        max_selections=min(12, len(other)),
        key="csv_series_pick",
    )

    for col in picks or []:
        st.caption(col)
        st.line_chart(
            [{c_time_name: row[c_time_name], col: row[col]} for row in chart_data],
            x=c_time_name,
            y=[col],
            height=220,
        )


# --- Streamlit 入口 -----------------------------------------------------------
st.set_page_config(
    page_title="RELAP5 参数优化系统",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
div[data-testid="stMetricValue"] { font-size: 1.35rem; }
.hero { opacity: 0.9; margin-top:-0.3rem;font-size:0.94rem;line-height:1.55; }
.stTabs [data-baseweb="tab-list"] { gap: 6px;}
</style>
""",
    unsafe_allow_html=True,
)


def init_dashboard_session() -> None:
    sd = {
        "running": False,
        "logs": [],
        "history": [],
        "snapshot_paths": {},
        "snapshot_dir": "",
        "run_bracket": (0.0, 0.0),
        "last_result": None,
        "worker_thread": None,
        "live_preview_thread": None,
        "live_preview_enabled": True,
        "live_preview_meta": None,
        "live_q": queue.Queue(),
        "stop_event": threading.Event(),
        "log_q": queue.Queue(),
        "iter_q": queue.Queue(),
        "result_q": queue.Queue(),
        "loaded_bundle": None,
        "analyze_csv_path": "",
        "dash_env_select": None,
    }
    for key, default in sd.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_dashboard_session()

env_select = st.sidebar.selectbox(
    "运行环境与配置文件",
    options=["dev", "test", "prod"],
    index=0,
    key="_sidebar_env_xyz",
)
effective_cfg = "prod" if env_select == "prod" else "dev"

st.sidebar.markdown(f"**基准目录**\n```\n{BASE_DIR}\n```")

if st.sidebar.button("↻ 从磁盘载入配置表单", disabled=st.session_state.running, use_container_width=True):
    st.session_state.loaded_bundle = None
    st.rerun()

st.sidebar.checkbox(
    f"实时预览（每 {int(LIVE_PREVIEW_INTERVAL_SEC)}s 从 rstplt 重算 p(t)）",
    key="live_preview_enabled",
    disabled=st.session_state.running,
    help=(
        "RELAP 评估往往要数分钟。开启后会在 plot/ 目录独立运行 strip.bat → "
        f"extract_plotrec_rows，写入 {LIVE_PREVIEW_CSV_REL}，"
        "「算法与运行」Tab 的实时面板会自动刷新。不影响优化主流程。"
    ),
)


def load_bundle_signature() -> str:
    """切换 dev/prod JSON 时使用。"""
    return f"{effective_cfg}"


if st.session_state.loaded_bundle != load_bundle_signature() and not st.session_state.running:
    hydrate_from_config(load_config_file(effective_cfg))
    st.session_state.loaded_bundle = load_bundle_signature()
    if not str(st.session_state.analyze_csv_path).strip():
        st.session_state.analyze_csv_path = st.session_state.get(
            "io_csv_rel",
            _default_integral_objective()["csv_relative_path"],
        )
drain_worker_queues()

if st.session_state.running:
    st.sidebar.markdown("### 运行状态")
    _live = (
        st.session_state.logs[-1]
        if st.session_state.logs
        else "已启动，等待 Python/RELAP 首条输出…"
    )
    st.sidebar.caption(_live if len(_live) < 480 else _live[:477] + "…")
    st.sidebar.caption("心跳约每 7s；本页 0.4s 刷新 · 切到「算法与运行」可看完整日志")

st.sidebar.caption("启动：`streamlit run optimization_dashboard.py`")

st.markdown("### RELAP5 · 可视化参数优化交互系统")
st.markdown(
    '<div class="hero">'
    '<b>参数与目标</b>填报 → <b>算法与运行</b>启动和监控 → '
    '<b>优化结果</b>分析 Pareto 与轨迹 → <b>数据可视化</b>对照 stripf/CSV。<br/>'
    f"配置文件：<code>config_{effective_cfg}.json</code>"
    "</div>",
    unsafe_allow_html=True,
)

t_save, t_run, t_stop, t_stat, _pad = st.columns((1.0, 1.15, 1.0, 1.05, 2.65))


def _save_clicked() -> None:
    try:
        cfg_to_save = build_runtime_config(env_select)
    except ValueError as exc:
        st.error(str(exc))
        return
    save_config_file(effective_cfg, cfg_to_save)
    st.toast(f"已保存 config_{effective_cfg}.json", icon="💾")


def _run_clicked() -> None:
    try:
        cfg = build_runtime_config(env_select)
    except ValueError as exc:
        st.error(str(exc))
        return
    start_worker(env=env_select, config=cfg)
    st.toast("优化任务已投递后台线程")


def _stop_clicked() -> None:
    st.session_state.stop_event.set()
    st.toast("停止信号已发送")


with t_save:
    if st.button("保存配置", use_container_width=True, disabled=st.session_state.running):
        _save_clicked()
with t_run:
    if st.button("启动优化", type="primary", use_container_width=True, disabled=st.session_state.running):
        _run_clicked()
with t_stop:
    if st.button("停止", use_container_width=True, disabled=not st.session_state.running):
        _stop_clicked()
with t_stat:
    st.metric("会话", "工作中" if st.session_state.running else "空闲")

tabs = st.tabs(["参数与目标", "算法与运行", "优化结果", "数据可视化"])

# ---- Tab 0 -----------------------------------------------------------------
with tabs[0]:
    with st.expander("定位 indta.i / CSV 列：字段含义摘录", expanded=False):
        st.markdown(
            """
| 项 | 说明 |
|--|--|
| line_key | 与 `indta.i` 中目标行第一个 token **完全一致**。 |
| column_index | 该行按空格分列、从 **0** 计数；若要改第 **4** 个数 ⇒ **3**。 |
| time_column_index | plotrec CSV 中作为时间轴的列索引（0 起）。 |
| value_column_index | plotrec CSV 中作为优化目标输出的列索引（0 起）。 |
"""
        )

    st.subheader("① 待优化参数（每行 = 1 个参数；可加/删行）")
    st.caption(
        "每个参数对应 indta.i 一行的某一列；多参数会同时在每次评估里写回。"
        "`line_key` 是行首 token，`column_index` 从 0 起。"
    )
    edited_params = st.data_editor(
        _format_param_rows_for_editor(st.session_state.get("params_rows", [])),
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn(
                "名称", required=True, help="标识用，可任意取名"
            ),
            "description": st.column_config.TextColumn("描述（可空）", width="medium"),
            "line_key": st.column_config.TextColumn(
                "line_key", required=True, help="indta.i 中行首 token"
            ),
            "column_index": st.column_config.NumberColumn(
                "col_idx", format="%d", help="0 起", step=1, required=True
            ),
            "initial_value": st.column_config.TextColumn(
                "initial_value", required=True,
                help="RELAP input card token, for example 10.0"
            ),
            "min_value": st.column_config.NumberColumn(
                "min_value", format="%.6g", required=True
            ),
            "max_value": st.column_config.NumberColumn(
                "max_value", format="%.6g", required=True
            ),
            "step": st.column_config.NumberColumn(
                "step", format="%.6g"
            ),
        },
        key="params_editor",
        hide_index=True,
        disabled=st.session_state.running,
    )
    st.session_state.params_rows = _format_param_rows_for_editor(
        edited_params if isinstance(edited_params, list)
        else (edited_params.to_dict("records")
              if hasattr(edited_params, "to_dict") else [])
    )

    st.subheader("② 优化目标（每行 = 1 个时间积分目标）")
    st.caption(
        "总目标 = ∑ weight × (∫|Δ|dt) / scale；`scale` 用于归一化不同量级的目标。"
    )
    edited_objs = st.data_editor(
        st.session_state.get("objectives_rows", []),
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("名称", required=True),
            "value_column_index": st.column_config.NumberColumn(
                "value_col_idx", format="%d", step=1, required=True,
                help="CSV 列索引（0 起）"
            ),
            "reference_value": st.column_config.NumberColumn(
                "reference_value", format="%g", required=True,
                help="希望 p(t) 接近的目标值"
            ),
            "deviation": st.column_config.SelectboxColumn(
                "deviation",
                options=list(DEVIATION_LABELS.keys()),
                required=True,
            ),
            "weight": st.column_config.NumberColumn(
                "weight", format="%g", required=True
            ),
            "scale": st.column_config.NumberColumn(
                "scale", format="%g", required=True,
                help="归一化因子；该目标 / scale 后再加权"
            ),
        },
        key="objectives_editor",
        hide_index=True,
        disabled=st.session_state.running,
    )
    st.session_state.objectives_rows = (
        edited_objs if isinstance(edited_objs, list)
        else (edited_objs.to_dict("records")
              if hasattr(edited_objs, "to_dict") else [])
    )

    st.subheader("③ 数据路径与时间列")
    aa, bb, cc = st.columns((1.0, 1.2, 0.75))
    aa.text_input("stripf 路径", key="io_stripf_rel", disabled=st.session_state.running)
    bb.text_input("CSV 路径", key="io_csv_rel", disabled=st.session_state.running)
    cc.number_input(
        "time_column_index",
        key="io_time_col",
        step=1,
        disabled=st.session_state.running,
    )

# ---- Tab 1 -----------------------------------------------------------------
with tabs[1]:
    st.subheader("求解器与运行参数")
    opt_l, opt_m, opt_r = st.columns((1.15, 0.9, 0.9))
    with opt_l:
        st.selectbox(
            "优化器",
            options=list(OPTIMIZER_LABELS.keys()),
            key="io_optimizer",
            format_func=lambda k: OPTIMIZER_LABELS[k],
            disabled=st.session_state.running,
            help=(
                "遗传算法是多目标默认选项；黄金分割仅适合单参数，"
                "多参数下主流程会自动切到贝叶斯优化。"
            ),
        )
    with opt_m:
        st.number_input(
            "max_function_evaluations",
            key="io_max_eval",
            step=1,
            min_value=1,
            disabled=st.session_state.running,
        )
    with opt_r:
        st.number_input(
            "max_relap_retries",
            key="io_max_relap_retries",
            step=1,
            min_value=0,
            disabled=st.session_state.running,
            help="单次评估内 RELAP 失败后向安全锚点二分回退并重跑的次数。",
        )

    alg_a, alg_b, alg_c, alg_d = st.columns(4)
    with alg_a:
        st.number_input(
            "convergence_tolerance",
            key="io_conv_tol",
            format="%g",
            disabled=st.session_state.running,
            help="仅黄金分割使用。",
        )
    with alg_b:
        st.number_input(
            "exploration_xi",
            key="io_exploration_xi",
            format="%g",
            disabled=st.session_state.running,
            help="仅贝叶斯优化使用。",
        )
    with alg_c:
        st.number_input(
            "random_seed",
            key="io_random_seed",
            step=1,
            disabled=st.session_state.running,
        )
    with alg_d:
        st.number_input(
            "max_iterations（兼容写盘）",
            key="max_iterations",
            step=1,
            min_value=1,
            disabled=st.session_state.running,
        )

    if st.session_state.get("io_optimizer") == "genetic":
        ga_l, ga_r = st.columns(2)
        with ga_l:
            st.number_input(
                "ga_crossover_probability",
                key="io_ga_crossover_probability",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format="%g",
                disabled=st.session_state.running,
            )
        with ga_r:
            st.number_input(
                "ga_mutation_scale",
                key="io_ga_mutation_scale",
                min_value=0.0,
                step=0.01,
                format="%g",
                disabled=st.session_state.running,
                help="相对于每个参数搜索区间宽度的高斯变异尺度。",
            )

    st.divider()
    lr_opt: Optional[Dict[str, Any]] = st.session_state.last_result

    live = compute_live_stats(st.session_state.history)
    fv_final = lr_opt.get("final_parameter_value") if lr_opt else None
    fo_final = lr_opt.get("final_objective_value") if lr_opt else None
    finite_fo_final = isinstance(fo_final, (int, float)) and math.isfinite(float(fo_final))

    mx = st.columns(5)
    with mx[0]:
        status = (
            "成功"
            if lr_opt and lr_opt.get("success")
            else ("运行中 …" if st.session_state.running else "—")
        )
        sr = lr_opt.get("stop_reason", "") if lr_opt else ""
        st.metric(
            "完成状态",
            status,
            delta=sr.replace("_", " ") or None if sr else None,
            delta_color="inverse" if lr_opt and not lr_opt.get("success") else "off",
        )

    # 历史最优目标（live cum-best）：运行中也实时刷新
    with mx[1]:
        live_best = live["best_obj"]
        if live_best is not None:
            st.metric(
                "最优目标值",
                f"{live_best:.6g}",
                delta=f"#{live['best_eval_id']}",
                delta_color="off",
                help="所有已完成评估中目标值（越小越好）的最小值；evaluation 编号见 delta",
            )
        elif finite_fo_final:
            st.metric("最优目标值", f"{float(fo_final):.6g}")
        else:
            st.metric("最优目标值", "—")

    # 最优参数值（取到 best_obj 的那个参数值）
    with mx[2]:
        live_param = live["best_param"]
        if live_param is not None:
            try:
                st.metric("最优参数值", f"{float(live_param):.6g}")
            except (TypeError, ValueError):
                st.metric("最优参数值", str(live_param))
        elif fv_final is not None:
            st.metric("最优参数值", f"{fv_final:.6g}")
        else:
            st.metric("最优参数值", "—")

    # 最近评估目标 + 与"上一个历史最优"的差值（负=改进，正=退步）
    with mx[3]:
        latest = live["latest_obj"]
        if latest is None:
            label = "+inf 或失败" if live["total"] > 0 else "—"
            st.metric("最近评估目标", label)
        else:
            d = live["delta_vs_prev_best"]
            if d is None:
                delta_str = "首次评估"
                d_color = "off"
            elif d < -1e-12:
                delta_str = f"-{abs(d):.4g}"
                d_color = "inverse"  # 负数 → 绿色（改进）
            elif d > 1e-12:
                delta_str = f"+{d:.4g}"
                d_color = "inverse"  # 正数 → 红色（变差）
            else:
                delta_str = "= 持平"
                d_color = "off"
            st.metric(
                "最近评估目标",
                f"{latest:.6g}",
                delta=delta_str,
                delta_color=d_color,
                help="本次评估目标值 · delta 为「与上一个历史最优」之差（负=改进 / 正=退步）",
            )

    # 已评估次数（含失败计数）
    with mx[4]:
        sfx = "评估"
        if live["total"] > 0:
            st.metric(
                f"已完成{sfx}",
                f"{live['total']}",
                delta=(f"{live['failed']} 失败" if live["failed"] else "全部成功"),
                delta_color=("inverse" if live["failed"] else "normal"),
            )
        else:
            st.metric(f"已完成{sfx}", "—")

    if st.session_state.history:
        latest_rec = st.session_state.history[-1]
        state_cols = st.columns((0.8, 1.45, 0.85, 0.9))
        with state_cols[0]:
            st.metric("当前评估", f"#{eval_id_of(latest_rec)}")
        with state_cols[1]:
            st.metric("参数向量", format_parameter_vector(latest_rec))
        with state_cols[2]:
            relap_state = "失败" if latest_rec.get("relap_failed") else "成功"
            if latest_rec.get("post_run_error"):
                relap_state = "后处理失败"
            st.metric("RELAP 状态", relap_state)
        with state_cols[3]:
            st.metric("失败重试次数", int(latest_rec.get("retries") or 0))

    if lr_opt and lr_opt.get("pareto_front"):
        st.markdown("##### Pareto 前沿")
        pareto_rows = flatten_pareto_front(lr_opt.get("pareto_front", []))
        st.dataframe(pareto_rows, hide_index=True, height=220)

    hist_rows = history_to_chart_rows(st.session_state.history)

    # ---- 偏差时间积分汇总（仅 minimize_time_integral 模式可见） ----
    is_integral_mode = st.session_state.get("objective") == "minimize_time_integral"
    if is_integral_mode:
        st.markdown(
            "##### 偏差时间积分 ∫|p(t) − p_ref|dt · 每次成功评估"
        )
        integral_rows = extract_integrals(st.session_state.history)
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.metric(
                "成功评估次数",
                f"{len(integral_rows)}",
                help="跳过失败 / +inf / NaN 的评估",
            )
        with ic2:
            best_int = min((r["integral"] for r in integral_rows), default=None)
            st.metric(
                "当前最小积分",
                "—" if best_int is None else f"{best_int:.6g}",
                help="∫|p(t) − p_ref|dt 当前历史最小值",
            )
        with ic3:
            if integral_rows:
                last = integral_rows[-1]
                st.metric(
                    f"最近成功（#{last['eval_id']}）",
                    f"{last['integral']:.6g}",
                    delta=("当前最优 ★" if last["is_best"] else "未刷新最优"),
                    delta_color=("normal" if last["is_best"] else "off"),
                )
            else:
                st.metric("最近成功", "—")

        if integral_rows:
            st.plotly_chart(
                build_integral_bar_figure(st.session_state.history),
                use_container_width=True,
                key="fig_integral_bar",
            )
            st.dataframe(
                integral_rows[-100:],
                hide_index=True,
                column_config={
                    "eval_id": st.column_config.NumberColumn("评估号", format="%d"),
                    "parameter": st.column_config.NumberColumn(
                        "参数值", format="%.6g"
                    ),
                    "integral": st.column_config.NumberColumn(
                        "∫|Δ|dt", format="%.6g",
                        help="本次评估的「优化变量与目标值偏差绝对值」对时间的积分",
                    ),
                    "is_best": st.column_config.CheckboxColumn(
                        "刷新最优", help="该次评估是否打破了之前的历史最小积分"
                    ),
                    "retries": st.column_config.NumberColumn(
                        "RELAP 重试", format="%d"
                    ),
                },
                height=210,
            )
        else:
            st.info(
                "尚无成功评估的积分记录。完成首次成功 RELAP 后，"
                "本面板会出现柱状图与表格。"
            )

    if st.session_state.running:
        tail_msg = (
            st.session_state.logs[-1]
            if st.session_state.logs
            else "任务已提交，正在初始化（若 RELAP 需数分钟，请先看到下方心跳日志）…"
        )
        st.info(f"**实时阶段** `{tail_msg}`")

    if st.session_state.running:
        st.caption(
            "运行中：界面约 **每 0.4s** 刷新；长时间 **start.bat / strip.bat** 会通过 **心跳行**（`[心跳 …]`）推送进度，"
            "请在本页观察「实时阶段」与右侧日志尾部。"
        )

    log_col, viz_col = st.columns((1.05, 0.98))
    with log_col:
        logs_text = "\n".join(st.session_state.logs[-900:])
        run_note = " （运行中滚动到底部可看最新心跳）" if st.session_state.running else ""
        st.caption(f"日志（尾 900 行）{run_note}")
        st.text_area("__logs__", value=logs_text, height=390, label_visibility="collapsed")
    with viz_col:
        if hist_rows:
            hx = chart_rows_with_x(hist_rows)
            st.caption("参数值随评估号的轨迹")
            st.line_chart(hx, x="_x", y=["parameter_value"], height=160)

            st.caption("**目标值收敛**：每次评估目标 + 历史最优（cum-best）")
            st.plotly_chart(
                build_convergence_figure(st.session_state.history, log_y=False),
                use_container_width=True,
                key="fig_convergence_inline",
            )
            st.download_button(
                "导出本轮迭代 CSV",
                data=history_to_download_csv(st.session_state.history),
                file_name=f"opt_trace_{effective_cfg}_{int(time.time())}.csv",
                mime="text/csv",
                key="csv_export_btn",
                use_container_width=True,
            )
            st.dataframe(st.session_state.history[-150:], hide_index=False, height=220)
        else:
            st.info("尚无评估记录。**启动优化** 后曲线与表格将逐步出现。")

    st.markdown("##### 实时计算预览 · `rstplt → plot/strip.bat → extract_plotrec_rows`")
    live_csv_abs = os.path.join(BASE_DIR, LIVE_PREVIEW_CSV_REL)
    live_meta = st.session_state.live_preview_meta
    live_exists = os.path.isfile(live_csv_abs)
    live_updated_ts: Optional[float] = None
    live_rows = 0
    live_meta_matches = False
    if isinstance(live_meta, dict):
        meta_path = str(live_meta.get("csv_path") or "")
        live_meta_matches = (
            bool(meta_path)
            and os.path.abspath(meta_path) == os.path.abspath(live_csv_abs)
        )
    if isinstance(live_meta, dict) and live_meta_matches:
        live_updated_ts = live_meta.get("updated_at")
        live_rows = int(live_meta.get("rows") or 0)
    if live_exists and not live_updated_ts:
        try:
            live_updated_ts = os.path.getmtime(live_csv_abs)
        except OSError:
            live_updated_ts = None
    if live_exists and not live_rows:
        live_rows = count_csv_data_rows(live_csv_abs)

    live_status_cols = st.columns(4)
    with live_status_cols[0]:
        st.metric(
            "实时预览",
            "已开启" if st.session_state.live_preview_enabled else "已关闭",
            delta=("运行中" if st.session_state.running else None),
            delta_color="off",
        )
    with live_status_cols[1]:
        st.metric("CSV 时间步数", "—" if not live_rows else f"{live_rows}")
    with live_status_cols[2]:
        ts_text = (
            time.strftime("%H:%M:%S", time.localtime(live_updated_ts))
            if live_updated_ts
            else "—"
        )
        st.metric("最近刷新", ts_text)
    with live_status_cols[3]:
        st.metric(
            "周期", f"{int(LIVE_PREVIEW_INTERVAL_SEC)}s",
            delta="独立 plot/" if st.session_state.running else None,
            delta_color="off",
        )

    if not live_exists:
        st.info(
            "尚未生成实时预览 CSV。**启动优化** 后约 "
            f"{int(LIVE_PREVIEW_INTERVAL_SEC)}s 内会出现 "
            f"`{LIVE_PREVIEW_CSV_REL}`。"
        )
    else:
        st.plotly_chart(
            build_live_pt_figure(
                csv_path=live_csv_abs,
                time_col=int(st.session_state.io_time_col),
                value_col=int(st.session_state.io_value_col),
                reference_value=float(st.session_state.io_reference),
                last_updated_at=live_updated_ts,
            ),
            use_container_width=True,
            key="fig_live_pt_inline",
        )
        st.caption(
            f"读取自 `{LIVE_PREVIEW_CSV_REL}` · "
            f"X = 第 {int(st.session_state.io_time_col)} 列（时间）· "
            f"Y = 第 {int(st.session_state.io_value_col)} 列 · "
            f"p_ref = {float(st.session_state.io_reference):g}"
        )

# ---- Tab 2: 优化结果（Plotly） ---------------------------------------------
with tabs[2]:
    snaps: Dict[int, str] = dict(st.session_state.get("snapshot_paths", {}))
    history_for_trace: List[Dict[str, Any]] = list(st.session_state.history)
    result_for_trace: Optional[Dict[str, Any]] = st.session_state.last_result

    if result_for_trace:
        st.subheader("代表最优解")
        best_cols = st.columns(4)
        with best_cols[0]:
            st.metric(
                "完成状态",
                "成功" if result_for_trace.get("success") else "未成功",
                delta=str(result_for_trace.get("stop_reason", "")).replace("_", " ") or None,
                delta_color="off",
            )
        with best_cols[1]:
            final_obj = result_for_trace.get("final_objective_value")
            st.metric(
                "加权目标值",
                "—" if final_obj is None else f"{float(final_obj):.6g}",
            )
        with best_cols[2]:
            final_vals = result_for_trace.get("final_parameter_values") or []
            st.metric(
                "最终参数向量",
                format_parameter_vector({"parameter_values": final_vals}),
            )
        with best_cols[3]:
            pareto_count = len(result_for_trace.get("pareto_front", []) or [])
            st.metric("Pareto 解数量", f"{pareto_count}")

        final_parts = result_for_trace.get("final_objective_values") or []
        if final_parts:
            st.dataframe(
                [
                    {"objective": f"objective_{i + 1}", "integral": v}
                    for i, v in enumerate(final_parts)
                ],
                hide_index=True,
                height=140,
            )

        if result_for_trace.get("pareto_front"):
            st.markdown("##### Pareto 前沿（按加权目标排序）")
            pareto_rows = flatten_pareto_front(result_for_trace.get("pareto_front", []))
            st.dataframe(pareto_rows, hide_index=True, height=240)
            st.download_button(
                "导出 Pareto CSV",
                data=rows_to_download_csv(pareto_rows),
                file_name="pareto_front.csv",
                mime="text/csv",
                key="pareto_export_btn",
            )

    if not history_for_trace:
        st.info(
            "**优化结果** 等待评估数据。**启动优化** 后，每完成一次评估会出现："
            "① 收敛曲线；② 参数轴搜索散点（多参数时为每个参数各一张）；③ 二维参数热图（≥2 参数时，任选两轴）；④ p(t) 对比。"
        )
    else:
        all_eval_ids = sorted({eval_id_of(it) for it in history_for_trace})
        snap_ids = sorted(snaps.keys())

        head_l, head_r = st.columns((1.0, 1.4))
        with head_l:
            log_y = st.checkbox("收敛曲线 Y 轴对数尺度", value=False, key="trace_logy")
        with head_r:
            st.caption(
                f"已记录 **{len(history_for_trace)}** 次评估"
                + (
                    f" · {len(snap_ids)} 个 CSV 快照（{st.session_state.snapshot_dir or '—'}）"
                    if snap_ids
                    else "（暂无 CSV 快照；仅展示参数轴 / 收敛图）"
                )
            )

        st.plotly_chart(
            build_convergence_figure(history_for_trace, log_y=bool(log_y)),
            use_container_width=True,
            key="fig_convergence",
        )

        # N≥2 时给每个参数一张独立的搜索散点图；N=1 时用单图保留兼容
        param_rows_now = list(st.session_state.get("params_rows", []))
        if len(param_rows_now) >= 2:
            param_names = [str(r.get("name", f"param_{i+1}"))
                           for i, r in enumerate(param_rows_now)]
            param_bounds = [(float(r.get("min_value", 0.0)),
                             float(r.get("max_value", 1.0)))
                            for r in param_rows_now]
            sub_figs = build_per_parameter_search_figures(
                history_for_trace, param_names, param_bounds,
            )
            for i, fig in enumerate(sub_figs):
                st.plotly_chart(
                    fig, use_container_width=True,
                    key=f"fig_search_axis_p{i}",
                )

            st.markdown("##### 参数平面 · 目标热图（任选两轴）")
            st.caption(
                "色阶表示加权聚合目标；背景为基于已有评估点的 IDW 插值近似，"
                "其余维度随优化历史变化——用于直观观察两维投影上的「好的区域」，而非严格 Pareto 曲面。"
            )
            n_hm = len(param_names)
            hm_c1, hm_c2, hm_c3 = st.columns((1.1, 1.1, 1.0))
            with hm_c1:
                hm_axis_i = st.selectbox(
                    "横轴参数",
                    options=list(range(n_hm)),
                    format_func=lambda i: f"{int(i) + 1}. {param_names[int(i)]}",
                    key="trace_hm_axis_i",
                )
            with hm_c2:
                hm_opts_j = [j for j in range(n_hm) if j != int(hm_axis_i)]
                hm_axis_j = st.selectbox(
                    "纵轴参数",
                    options=hm_opts_j,
                    format_func=lambda j: f"{int(j) + 1}. {param_names[int(j)]}",
                    key="trace_hm_axis_j",
                )
            with hm_c3:
                hm_grid = st.slider(
                    "热图网格分辨率",
                    min_value=16,
                    max_value=72,
                    value=40,
                    step=4,
                    key="trace_hm_grid_res",
                )
            st.plotly_chart(
                build_parameter_pair_objective_heatmap(
                    history_for_trace,
                    param_names,
                    param_bounds,
                    axis_i=int(hm_axis_i),
                    axis_j=int(hm_axis_j),
                    grid_resolution=int(hm_grid),
                ),
                use_container_width=True,
                key="fig_param_pair_heatmap",
            )
        else:
            st.plotly_chart(
                build_search_axis_figure(
                    history_for_trace,
                    bracket=tuple(st.session_state.run_bracket or (0.0, 0.0)),
                ),
                use_container_width=True,
                key="fig_search_axis",
            )

        st.markdown("##### p(t) 时间序列对比 · 参考线 · 偏差填充区")
        if not snap_ids:
            st.warning(
                "尚未捕获任何 CSV 快照。仅 `minimize_time_integral` 模式且 RELAP 成功生成 plotrec CSV 时才会保存。"
            )
        else:
            default_sel = snap_ids if len(snap_ids) <= 4 else [
                snap_ids[0],
                snap_ids[len(snap_ids) // 2],
                snap_ids[-1],
            ]
            ctl_l, ctl_r = st.columns((1.4, 1.0))
            with ctl_l:
                sel = st.multiselect(
                    "选择对比的评估号",
                    options=snap_ids,
                    default=default_sel,
                    key="trace_compare_sel",
                )
            with ctl_r:
                focus_options = sel or snap_ids
                focus = st.selectbox(
                    "聚焦评估（高亮 + 偏差填充）",
                    options=focus_options,
                    index=len(focus_options) - 1,
                    key="trace_focus_sel",
                )

            time_col_idx = int(st.session_state.io_time_col)
            value_col_idx = int(st.session_state.io_value_col)
            ref_v = float(st.session_state.io_reference)

            st.plotly_chart(
                build_trace_compare_figure(
                    history_for_trace,
                    snapshots=snaps,
                    time_col=time_col_idx,
                    value_col=value_col_idx,
                    reference_value=ref_v,
                    selected_eval_ids=list(sel) if sel else None,
                    focus_eval_id=int(focus) if focus is not None else None,
                ),
                use_container_width=True,
                key="fig_trace_compare",
            )

            st.caption(
                f"X = CSV 第 {time_col_idx} 列（时间）·  Y = 第 {value_col_idx} 列（输出参数）·  "
                f"参考线 p_ref = {ref_v:g}（取自配置 `integral_objective.reference_value`）"
            )

# ---- Tab 3: 数据可视化 ------------------------------------------------------
with tabs[3]:
    st.markdown("###### Plotrec CSV 交互曲线")
    pth = st.text_input(
        "CSV 路径（空白则使用表单中的 CSV 字段）",
        key="analyze_csv_path",
        help=f"项目根：**{BASE_DIR}**",
    )
    csv_for_viz = pth.strip() if pth.strip() else str(st.session_state.io_csv_rel)
    render_csv_charts(csv_for_viz)

if st.session_state.running:
    # 更短周期便于在 RELAP 阻塞期间尽快显示 run_bat 心跳日志
    time.sleep(0.4)
    st.rerun()
