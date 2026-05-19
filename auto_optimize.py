import argparse
import json
import locale
import os
import subprocess
import sys
import math
import threading
import time as time_std
from typing import List, Dict, Any, Tuple, Callable, Optional, Sequence

from bayesian_optimizer import bo_minimize, bo_minimize_nd
from genetic_optimizer import ga_minimize_pareto
from integral_objective import (
    compute_time_integral_objective,
    compute_weighted_objective,
    golden_section_minimize,
)
from live_preview import refresh_live_csv
from outdta_check import outdta_failure_diagnosis

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDTA_O_REL = "outdta.o"


def check_outdta_failed(emit: Callable[[str], None]) -> Dict[str, Any]:
    """
    检查 ``outdta.o`` 末行是否带有 RELAP 失败签名。

    返回 ``outdta_failure_diagnosis`` 字典；同时通过 ``emit`` 输出可读日志。
    在 RELAP 末行 ``Transient terminated by ...`` 通常是计算结局的权威标记，
    见 :data:`outdta_check.OUTDTA_FAILURE_PATTERNS`。
    """
    diag = outdta_failure_diagnosis(os.path.join(BASE_DIR, OUTDTA_O_REL))
    if not diag["exists"]:
        emit(f"  ! 未发现 {OUTDTA_O_REL}（无法判断 RELAP 是否成功）")
        return diag
    if diag["failed"]:
        emit(
            f"  ✗ RELAP 失败（命中模式：{diag['matched_pattern']!r}）\n"
            f"     {OUTDTA_O_REL} 末行: {diag['last_line']}"
        )
    else:
        emit(f"  ✓ RELAP 末行: {diag['last_line']}")
    return diag


def _compute_recovery_candidate(
    failed_x: Any,
    last_safe_x: Any,
    initial_value: Any,
    bracket: Any,
    min_step: float = 1e-9,
) -> Any:
    """
    给定一次 RELAP 在 ``failed_x`` 处失败，返回下一个建议尝试的参数（标量或向量）。

    - **1D（标量）**：``failed_x``、``last_safe_x``、``initial_value`` 是 float；
      ``bracket = (a, b)``。返回标量或 ``None``。
    - **N-D（向量）**：``failed_x``、``last_safe_x``、``initial_value`` 是
      ``ndarray[d]`` / ``Sequence[float]``；``bracket = [(a₁,b₁), …, (a_d,b_d)]``。
      返回 ``ndarray[d]`` 或 ``None``。

    "向已知安全区域二分回退"，锚点优先级：

    1. ``last_safe_x``（若不为 None 且与 ``failed_x`` 距离 > ``min_step``）
    2. ``initial_value``
    3. ``bracket`` 中心点

    返回值会被 clamp 到 ``bracket`` 内。所有锚点都与 ``failed_x`` 重合时返回 None。
    """
    # 通过 bracket 形态判断维度
    is_1d = (
        isinstance(bracket, tuple)
        and len(bracket) == 2
        and isinstance(bracket[0], (int, float))
    )
    if is_1d:
        return _recovery_candidate_1d(
            float(failed_x),
            None if last_safe_x is None else float(last_safe_x),
            float(initial_value),
            (float(bracket[0]), float(bracket[1])),
            min_step,
        )
    # N-D
    bounds = np.asarray(bracket, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bracket must be (a,b) or [(a,b),...]; got {bracket!r}")
    fx = np.asarray(failed_x, dtype=float).flatten()
    if fx.shape[0] != bounds.shape[0]:
        raise ValueError(
            f"failed_x 长度 {fx.shape[0]} 与 bracket 维度 {bounds.shape[0]} 不一致"
        )

    anchor = None
    if last_safe_x is not None:
        cand = np.asarray(last_safe_x, dtype=float).flatten()
        if cand.shape == fx.shape and float(np.linalg.norm(cand - fx)) > min_step:
            anchor = cand
    if anchor is None:
        cand = np.asarray(initial_value, dtype=float).flatten()
        if cand.shape == fx.shape and float(np.linalg.norm(cand - fx)) > min_step:
            anchor = cand
    if anchor is None:
        cand = (bounds[:, 0] + bounds[:, 1]) / 2.0
        if float(np.linalg.norm(cand - fx)) > min_step:
            anchor = cand
    if anchor is None:
        return None

    new_x = (fx + anchor) / 2.0
    new_x = np.maximum(bounds[:, 0], np.minimum(bounds[:, 1], new_x))
    if float(np.linalg.norm(new_x - fx)) < min_step:
        return None
    return new_x


def _recovery_candidate_1d(
    failed_x: float,
    last_safe_x: Optional[float],
    initial_value: float,
    bracket: Tuple[float, float],
    min_step: float,
) -> Optional[float]:
    """1D 标量版本（保留旧测试契约）。"""
    a, b = float(bracket[0]), float(bracket[1])
    fx = float(failed_x)

    candidates: List[float] = []
    if last_safe_x is not None and abs(float(last_safe_x) - fx) > min_step:
        candidates.append(float(last_safe_x))
    if abs(float(initial_value) - fx) > min_step:
        candidates.append(float(initial_value))
    mid = (a + b) / 2.0
    if abs(mid - fx) > min_step:
        candidates.append(mid)

    if not candidates:
        return None

    anchor = candidates[0]
    new_x = (fx + anchor) / 2.0
    new_x = max(a, min(b, new_x))
    if abs(new_x - fx) < min_step:
        return None
    return new_x


def _vec_or_scalar_to_str(x: Any) -> str:
    """简洁地把标量或向量参数渲染成日志可读字符串。"""
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if arr.size == 1:
        return format_value(float(arr[0]))
    return "[" + ", ".join(format_value(float(v)) for v in arr) + "]"


def _try_relap_with_recovery(
    x_request: Any,
    *,
    last_safe_x: Any,
    initial_value: Any,
    bracket: Any,
    max_retries: int,
    indta_path: str,
    param_cfgs: Sequence[Dict[str, Any]],
    emit: Callable[[str], None],
    stop_requested: Optional[Callable[[], bool]] = None,
) -> Tuple[np.ndarray, Dict[str, Any], List[np.ndarray]]:
    """
    在 ``x_request`` 处尝试一次 RELAP；失败时按 :func:`_compute_recovery_candidate`
    向安全锚点二分回退，最多 ``max_retries`` 次重调，直到不再失败或穷尽尝试。

    Parameters
    ----------
    x_request, last_safe_x, initial_value
        标量（1 个参数时）或 ``ndarray[d]`` / 序列（多参数时）
    bracket
        ``(a, b)`` 标量元组（1 个参数时）或 ``[(a₁,b₁), …]`` （N 维时）
    param_cfgs
        参数配置**列表**（即使只有 1 个参数也是单元素列表）；长度必须与
        ``x_request`` 的维度一致

    Returns
    -------
    ``(candidate_used: ndarray[d], diag, attempted_values: List[ndarray[d]])``
    始终用向量形式返回；上层若是 1D 直接取 ``[0]`` 即可。
    """
    candidate = np.atleast_1d(np.asarray(x_request, dtype=float)).copy()
    if candidate.shape[0] != len(param_cfgs):
        raise ValueError(
            f"x_request 长度 {candidate.shape[0]} 与 param_cfgs 数量 {len(param_cfgs)} 不一致"
        )
    attempted: List[np.ndarray] = []
    diag: Dict[str, Any] = {"failed": False, "last_line": "", "matched_pattern": None}

    total_max = max(0, int(max_retries)) + 1  # 1 次首跑 + N 次重试

    for attempt in range(total_max):
        if stop_requested is not None and stop_requested():
            diag = {"failed": True, "last_line": "stopped_by_user", "matched_pattern": None}
            return candidate, diag, attempted

        attempted.append(candidate.copy())

        lines = load_indta(indta_path)
        new_lines = update_parameters_in_indta(lines, param_cfgs, candidate.tolist())
        save_indta(indta_path, new_lines)

        cand_str = _vec_or_scalar_to_str(candidate)
        if attempt == 0:
            emit(f"  Running start.bat ...（首次尝试 · parameters = {cand_str}）")
        else:
            emit(f"  Running start.bat ...（重试 #{attempt} · parameters = {cand_str}）")
        run_bat(
            "start.bat",
            emit=emit,
            expect_file_after="outdta",
            phase_label="主工况 start.bat",
        )

        diag = check_outdta_failed(emit)
        if not diag["failed"]:
            return candidate, diag, attempted

        if attempt + 1 >= total_max:
            emit(f"  ✗ 已达最大重试次数 {max_retries}；停止重调。")
            return candidate, diag, attempted

        next_c = _compute_recovery_candidate(
            failed_x=candidate,
            last_safe_x=last_safe_x,
            initial_value=initial_value,
            bracket=bracket,
        )
        if next_c is None:
            emit("  ✗ 无可回退的安全锚点（current 已与 last_safe / initial / 区间中点重合）；停止重调。")
            return candidate, diag, attempted

        # 决定锚点标签（仅作日志）
        ls = None
        if last_safe_x is not None:
            ls = np.atleast_1d(np.asarray(last_safe_x, dtype=float))
            if ls.shape != candidate.shape or float(np.linalg.norm(ls - candidate)) <= 1e-9:
                ls = None
        if ls is not None:
            anchor_label = "last_safe"
        else:
            iv = np.atleast_1d(np.asarray(initial_value, dtype=float))
            if iv.shape == candidate.shape and float(np.linalg.norm(iv - candidate)) > 1e-9:
                anchor_label = "initial_value"
            else:
                anchor_label = "bracket_mid"

        emit(
            f"  → 重新调整参数：{cand_str} → {_vec_or_scalar_to_str(next_c)}"
            f"（向 {anchor_label} 方向二分回退）"
        )
        candidate = np.atleast_1d(np.asarray(next_c, dtype=float)).copy()

    return candidate, diag, attempted


def load_config(env: str) -> Dict[str, Any]:
    if env == "prod":
        config_path = os.path.join(BASE_DIR, "config_prod.json")
    else:
        config_path = os.path.join(BASE_DIR, "config_dev.json")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_indta(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def save_indta(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def format_value(value: float) -> str:
    # RELAP input cards expect floating-point tokens to include a decimal point.
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"RELAP parameter value must be finite: {value!r}")

    text = f"{v:.6g}"
    if "e" in text.lower():
        mantissa, exponent = text.lower().split("e", 1)
        if "." not in mantissa:
            mantissa += ".0"
        return f"{mantissa}e{exponent}"
    if "." not in text:
        text += ".0"
    return text


def normalize_parameters_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    把配置里的 parameters/parameter（多/单数）统一成 ``List[Dict]``。

    顺序：
      1. ``config["parameters"]``（list）—— 新格式，直接返回
      2. ``config["parameter"]``（dict）—— 旧格式，包成单元素列表
      3. 否则抛 ``KeyError``
    """
    if "parameters" in config:
        params = config["parameters"]
        if not isinstance(params, list) or not params:
            raise ValueError("parameters 必须是非空列表")
        return list(params)
    if "parameter" in config:
        return [dict(config["parameter"])]
    raise KeyError("配置中未找到 parameters / parameter")


def normalize_objectives_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    把配置里的 objectives/integral_objective（多/单数）统一成 ``List[Dict]``。

    顺序：
      1. ``config["objectives"]``（list）—— 新格式，直接返回
      2. ``config["integral_objective"]``（dict）—— 旧格式，从中拷出 value_column_index/
         reference_value/deviation 包成单元素列表，自动补默认 weight=1, scale=1
      3. 否则抛 ``KeyError``
    """
    if "objectives" in config:
        objs = config["objectives"]
        if not isinstance(objs, list) or not objs:
            raise ValueError("objectives 必须是非空列表")
        return list(objs)
    if "integral_objective" in config:
        io = config["integral_objective"]
        return [{
            "name": io.get("name", "integral"),
            "value_column_index": int(io["value_column_index"]),
            "reference_value": float(io["reference_value"]),
            "deviation": str(io.get("deviation", "absolute")),
            "weight": 1.0,
            "scale": 1.0,
        }]
    raise KeyError("配置中未找到 objectives / integral_objective")


def validate_integral_optimization_config(config: Dict[str, Any]) -> None:
    """
    集中校验多参数、多目标时间积分优化配置。

    该函数只验证配置契约，不读取/写入 RELAP 输入输出文件；调用方可在 UI 启动前
    或命令行入口处提前失败，避免把非法配置送进长时间计算。
    """
    try:
        params = normalize_parameters_config(config)
        objs = normalize_objectives_config(config)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"配置校验失败：{exc}") from exc
    int_cfg = config.get("integral_objective", {})
    opt_cfg = config.get("integral_optimizer", {})

    errors: List[str] = []

    for i, p in enumerate(params):
        label = str(p.get("name") or f"parameters[{i}]")
        if not str(p.get("line_key", "")).strip():
            errors.append(f"{label}: line_key 不能为空")
        try:
            col = int(p["column_index"])
            if col < 0:
                errors.append(f"{label}: column_index 不能为负数")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}: column_index 必须是整数")

        try:
            initial = float(p["initial_value"])
            lo = float(p["min_value"])
            hi = float(p["max_value"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}: initial/min/max 必须是数值")
            continue
        if not all(math.isfinite(v) for v in (initial, lo, hi)):
            errors.append(f"{label}: initial/min/max 必须是有限数")
            continue
        if lo >= hi:
            errors.append(f"{label}: min_value 必须小于 max_value")
        if initial < lo or initial > hi:
            errors.append(f"{label}: initial_value 必须位于 [min_value, max_value] 内")

    for i, obj in enumerate(objs):
        label = str(obj.get("name") or f"objectives[{i}]")
        try:
            col = int(obj["value_column_index"])
            if col < 0:
                errors.append(f"{label}: value_column_index 不能为负数")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}: value_column_index 必须是整数")
        try:
            ref = float(obj["reference_value"])
            weight = float(obj.get("weight", 1.0))
            scale = float(obj.get("scale", 1.0))
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}: reference_value/weight/scale 必须是数值")
            continue
        if not all(math.isfinite(v) for v in (ref, weight, scale)):
            errors.append(f"{label}: reference_value/weight/scale 必须是有限数")
        if scale == 0.0:
            errors.append(f"{label}: scale 不能为 0")
        deviation = str(obj.get("deviation", "absolute"))
        if deviation not in ("absolute", "signed", "squared"):
            errors.append(f"{label}: deviation 必须是 absolute/signed/squared")

    try:
        time_col = int(int_cfg.get("time_column_index", 0))
        if time_col < 0:
            errors.append("integral_objective.time_column_index 不能为负数")
    except (TypeError, ValueError):
        errors.append("integral_objective.time_column_index 必须是整数")

    try:
        max_eval = int(opt_cfg.get("max_function_evaluations", 30))
        if max_eval <= 0:
            errors.append("integral_optimizer.max_function_evaluations 必须大于 0")
    except (TypeError, ValueError):
        errors.append("integral_optimizer.max_function_evaluations 必须是整数")

    try:
        max_retries = int(opt_cfg.get("max_relap_retries", 10))
        if max_retries < 0:
            errors.append("integral_optimizer.max_relap_retries 不能为负数")
    except (TypeError, ValueError):
        errors.append("integral_optimizer.max_relap_retries 必须是整数")

    optimizer = str(opt_cfg.get("optimizer", "genetic")).lower()
    if optimizer not in ("golden_section", "bayesian", "genetic"):
        errors.append("integral_optimizer.optimizer 必须是 golden_section/bayesian/genetic")

    try:
        crossover = float(opt_cfg.get("ga_crossover_probability", 0.9))
        if not 0.0 <= crossover <= 1.0:
            errors.append("integral_optimizer.ga_crossover_probability 必须位于 [0, 1]")
    except (TypeError, ValueError):
        errors.append("integral_optimizer.ga_crossover_probability 必须是数值")

    try:
        mutation = float(opt_cfg.get("ga_mutation_scale", 0.12))
        if mutation < 0.0 or not math.isfinite(mutation):
            errors.append("integral_optimizer.ga_mutation_scale 必须是非负有限数")
    except (TypeError, ValueError):
        errors.append("integral_optimizer.ga_mutation_scale 必须是数值")

    if errors:
        raise ValueError("配置校验失败：\n- " + "\n- ".join(errors))


def update_parameters_in_indta(
    lines: List[str],
    param_cfgs: Sequence[Dict[str, Any]],
    values: Sequence[float],
) -> List[str]:
    """
    一次把多个参数都写回 indta.i。内部就是循环复用 :func:`update_parameter_in_indta`，
    在每次迭代里读到的是「已应用前序更新」的 lines —— 当多个参数共享同一行时
    后写的会覆盖前面的相同列。
    """
    if len(param_cfgs) != len(values):
        raise ValueError(
            f"param_cfgs 与 values 长度不一致：{len(param_cfgs)} vs {len(values)}"
        )
    updated = lines
    for cfg, v in zip(param_cfgs, values):
        updated = update_parameter_in_indta(updated, cfg, float(v))
    return updated


def update_parameter_in_indta(
    lines: List[str], param_cfg: Dict[str, Any], new_value: float
) -> List[str]:
    line_key = str(param_cfg["line_key"]).strip()
    col_idx = int(param_cfg["column_index"])
    new_lines: List[str] = []

    for line in lines:
        # Split off any comments (starting with '*')
        prefix, sep, comment = line.partition("*")
        stripped = prefix.strip()
        if stripped:
            tokens = stripped.split()
        else:
            tokens = []

        if tokens and tokens[0] == line_key:
            # Ensure there is enough columns
            if len(tokens) <= col_idx:
                raise ValueError(
                    f"Line with key {line_key} does not have column index {col_idx}: {line!r}"
                )
            tokens[col_idx] = format_value(new_value)
            new_prefix = "  ".join(tokens)
            new_line = new_prefix
            if sep:  # there was a comment
                new_line += f" *{comment}"
            else:
                # Preserve original newline if present, otherwise add one
                if not new_line.endswith("\n"):
                    new_line += "\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    return new_lines


def extract_plotrec_csv(
    stripf_rel: str = "stripf",
    csv_rel: str = "plot/stripf_plotrec_row.csv",
) -> str:
    """
    Run ``extract_plotrec_rows.py`` to build CSV from stripf (after ``strip.bat``).

    默认输出路径与「实时预览」共用同一份 CSV：``plot/stripf_plotrec_row.csv``。
    通过 :data:`live_preview.CSV_WRITE_LOCK` 与实时预览线程互斥写入，避免文件交错。
    """
    stripf_path = os.path.join(BASE_DIR, stripf_rel)
    out_path = os.path.join(BASE_DIR, csv_rel)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    candidates = [
        os.path.join(BASE_DIR, "extract_plotrec_rows.py"),
        os.path.join(BASE_DIR, "plot", "extract_plotrec_rows.py"),
    ]
    script = next((c for c in candidates if os.path.exists(c)), "")
    if not script:
        raise FileNotFoundError(
            "extract_plotrec_rows.py not found in expected locations: "
            + ", ".join(candidates)
        )

    # 与实时预览共享 live.csv，必须串行写入（懒导入避免循环依赖；live_preview
    # 不可用时退化为无锁，行为与历史一致）。
    try:
        from live_preview import CSV_WRITE_LOCK as _lock
    except ImportError:
        _lock = None

    cmd = [sys.executable, script, "--input", stripf_path, "--output", out_path]
    if _lock is not None:
        with _lock:
            subprocess.run(cmd, cwd=BASE_DIR, check=True)
    else:
        subprocess.run(cmd, cwd=BASE_DIR, check=True)
    return out_path


def run_bat(
    script_name: str,
    emit: Optional[Callable[[str], None]] = None,
    expect_file_after: Optional[str] = None,
    phase_label: str = "",
    min_expected_bytes_after: int = 32,
    heartbeat_interval_sec: float = 7.0,
) -> float:
    """
    Run a .bat under BASE_DIR.

    Notes (Windows RELAP workflows)
    -------------------------------
    Many batch files continue with exit code **0 even when ``relap.exe`` fails**.
    Passing ``expect_file_after`` (path relative to BASE_DIR) rejects the bogus
    "instant success" path by requiring a non‑trivial output file after each phase.

    While ``subprocess.run`` blocks (often minutes for RELAP), a background
    **heartbeat** calls ``emit`` every ``heartbeat_interval_sec`` so UIs can
    refresh and show liveness.

    Returns wall‑clock seconds spent inside the subprocess.
    """
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"{script_name} not found at {script_path}")

    artifact = os.path.join(BASE_DIR, expect_file_after) if expect_file_after else ""
    artifact_mtime_before = (
        os.path.getmtime(artifact) if artifact and os.path.isfile(artifact) else None
    )

    label = phase_label or script_name
    if emit:
        hb = float(heartbeat_interval_sec)
        if hb > 0:
            emit(
                f"  → 正在执行 {script_name}（{label}），"
                f"阻塞型计算中约每 {hb:.0f}s 推送一次心跳…"
            )
        else:
            emit(f"  → 正在执行 {script_name}（{label}）…")

    stop_hb = threading.Event()

    def _heartbeat() -> None:
        t_start = time_std.perf_counter()
        interval = max(1.0, float(heartbeat_interval_sec))
        while not stop_hb.wait(timeout=interval):
            if emit:
                wall = time_std.perf_counter() - t_start
                emit(f"[心跳 {wall:,.0f}s] 仍在运行 {script_name} · {label}")

    hb_thread: Optional[threading.Thread] = None
    if emit and heartbeat_interval_sec and heartbeat_interval_sec > 0:
        hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        hb_thread.start()

    t0 = time_std.perf_counter()
    wall_start = time_std.time()
    completed: Optional[subprocess.CompletedProcess[str]] = None
    try:
        completed = subprocess.run(
            ["cmd.exe", "/c", "call", script_path],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding=locale.getpreferredencoding(False) or "utf-8",
            errors="replace",
            check=False,
        )
    finally:
        stop_hb.set()
        if hb_thread is not None:
            hb_thread.join(timeout=2.0)

    elapsed = time_std.perf_counter() - t0
    returncode = int(completed.returncode) if completed is not None else -1
    output = completed.stdout if completed is not None and completed.stdout else ""

    if emit and output:
        lines = [ln.rstrip() for ln in output.splitlines() if ln.strip()]
        max_lines = 40
        if len(lines) > max_lines:
            emit(
                f"  → {script_name} 输出较长，仅显示最后 {max_lines}/{len(lines)} 行："
            )
            lines = lines[-max_lines:]
        else:
            emit(f"  → {script_name} 输出：")
        for line in lines:
            emit(f"    {line}")

    if returncode != 0 and emit:
        emit(
            f"  [WARN] {script_name} 返回退出码 {returncode}；"
            "继续检查本次输出文件与 RELAP 终止状态。"
        )

    if expect_file_after:
        if not os.path.isfile(artifact):
            msg = (
                f"{phase_label or script_name}: 运行后未找到 `{expect_file_after}`。"
                " 通常为 relap.exe 未生成输出或.bat 没有在失败时 exit /b 1。"
                f" 子进程退出码 {returncode}。"
                f" （本次子进程耗时 {elapsed:.2f}s，若本应数分钟则可能未真正求解。）"
            )
            if emit:
                emit(msg)
            raise RuntimeError(msg)

        mtime = os.path.getmtime(artifact)
        if mtime < wall_start - 1.0 or (
            artifact_mtime_before is not None and mtime <= artifact_mtime_before
        ):
            msg = (
                f"{phase_label or script_name}: `{expect_file_after}` 未在本次运行中刷新，"
                f"可能仍是旧结果。子进程退出码 {returncode}，耗时 {elapsed:.2f}s。"
            )
            if emit:
                emit(msg)
            raise RuntimeError(msg)

        sz = os.path.getsize(artifact)
        if sz < min_expected_bytes_after:
            msg = (
                f"{phase_label or script_name}: `{expect_file_after}` 过小（{sz} 字节），"
                f"不像一次完整 RELAP 输出。子进程退出码 {returncode}，"
                f"耗时 {elapsed:.2f}s。"
            )
            if emit:
                emit(msg)
            raise RuntimeError(msg)
    elif returncode != 0:
        msg = f"{script_name} failed with exit code {returncode}"
        if emit:
            emit(msg)
        raise RuntimeError(msg)

    if emit:
        emit(f"  → {script_name} 结束，退出码 {returncode}，耗时 {elapsed:.2f}s")

    return elapsed


def _try_parse_float_tokens(tokens: List[str]) -> Optional[List[float]]:
    try:
        return [float(tok) for tok in tokens]
    except ValueError:
        return None


def parse_stripf_records(stripf_path: str) -> List[List[float]]:
    """
    Parse 'plotrec' blocks in stripf.

    In this deck, each logical record is written as:

      plotrec  <time>  <velf>  <voidg>  <voidg>
       <voidg> <voidg>

    i.e. the keyword 'plotrec' is on the first physical line, with up to four
    numbers following, and the remaining numbers for the same time step are on
    the next physical line (without the keyword).

    This function:
    - finds lines starting with 'plotrec'
    - parses the numbers on that line
    - reads the very next physical line and, if it contains additional
      numbers, appends them to the same record
    """
    records: List[List[float]] = []
    with open(stripf_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    i = 0
    total = len(raw_lines)
    while i < total:
        line = raw_lines[i].lstrip()
        if not line.startswith("plotrec"):
            i += 1
            continue

        # First physical line: remove keyword and parse numbers
        rest = line[len("plotrec") :].strip()
        if not rest:
            i += 1
            continue

        first_numbers = _try_parse_float_tokens(rest.split())
        if first_numbers is None:
            i += 1
            continue

        numbers = list(first_numbers)

        # Optional continuation line:
        # only append when the next line is plain numeric tokens
        # (not another record/header/keyword line).
        if i + 1 < total:
            next_line = raw_lines[i + 1].strip()
            if next_line and not next_line.lower().startswith("plotrec"):
                more_numbers = _try_parse_float_tokens(next_line.split())
                if more_numbers is not None:
                    numbers.extend(more_numbers)
                    i += 1

        records.append(numbers)
        i += 1

    return records


def get_last_record(stripf_path: str) -> List[float]:
    """
    Return the last full 'plotrec' record (all columns) from stripf.

    This keeps the original behavior for existing code that depends on the
    final time-step values.
    """
    records = parse_stripf_records(stripf_path)
    if not records:
        raise RuntimeError("No 'plotrec' records found in stripf.")
    return records[-1]


def get_column_series(stripf_path: str, column_index: int) -> List[float]:
    """
    Extract a time series of a specified column from all 'plotrec' records.

    Parameters
    ----------
    stripf_path : str
        Path to the stripf output file.
    column_index : int
        Zero-based index of the column to extract (0=time, 1=velf, 2..=voidg...).

    Returns
    -------
    List[float]
        The values of the requested column for all time steps, in order.
    """
    records = parse_stripf_records(stripf_path)
    if not records:
        raise RuntimeError("No 'plotrec' records found in stripf.")

    series: List[float] = []
    for i, rec in enumerate(records):
        if column_index < 0 or column_index >= len(rec):
            raise IndexError(
                f"Column index {column_index} out of range for record {i} "
                f"with length {len(rec)}"
            )
        series.append(rec[column_index])

    return series


def evaluate_criterion(
    record: List[float], crit_cfg: Dict[str, Any]
) -> Tuple[bool, float]:
    idx = int(crit_cfg["target_column_index"])
    if idx < 0 or idx >= len(record):
        raise IndexError(
            f"Criterion target_column_index {idx} out of range for record length {len(record)}"
        )
    value = record[idx]
    target = float(crit_cfg["target_value"])
    tol = float(crit_cfg["tolerance"])
    error = value - target
    success = abs(error) <= tol
    return success, error


def suggest_next_value(
    current: float, error: float, param_cfg: Dict[str, Any]
) -> float:
    step = float(param_cfg["step"])
    vmin = float(param_cfg["min_value"])
    vmax = float(param_cfg["max_value"])

    if error > 0:
        proposed = current - step
    elif error < 0:
        proposed = current + step
    else:
        proposed = current

    # Clamp to range
    if proposed < vmin:
        proposed = vmin
    if proposed > vmax:
        proposed = vmax
    return proposed


def run_optimization_minimize_time_integral(
    env: str = "dev",
    config_override: Optional[Dict[str, Any]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    on_iteration: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    多参数 + 多目标加权和最小化的优化主入口。

    后处理用 :func:`live_preview.refresh_live_csv` 走 ``plot/strip.bat`` 管线，
    目标函数为 :func:`integral_objective.compute_weighted_objective`：

        f(x) = Σᵢ wᵢ · (∫|pᵢ(t) − refᵢ| dt) / scaleᵢ

    向后兼容旧 schema：单数 ``parameter``/``integral_objective`` 会被
    :func:`normalize_parameters_config` / :func:`normalize_objectives_config`
    自动包装成单元素列表，对老配置文件零侵入。
    """
    config = config_override if config_override is not None else load_config(env)
    validate_integral_optimization_config(config)
    param_cfgs = normalize_parameters_config(config)
    obj_cfgs = normalize_objectives_config(config)
    int_cfg = config.get("integral_objective", {})
    opt_cfg = config.get("integral_optimizer", {})

    indta_path = os.path.join(BASE_DIR, "indta.i")
    stripf_rel = str(int_cfg.get("stripf_relative_path", "stripf"))
    csv_rel = str(
        int_cfg.get("csv_relative_path", "plot/stripf_plotrec_row.csv")
    )

    time_col = int(int_cfg.get("time_column_index", 0))

    n_params = len(param_cfgs)
    a_vec = np.array([float(p["min_value"]) for p in param_cfgs], dtype=float)
    b_vec = np.array([float(p["max_value"]) for p in param_cfgs], dtype=float)
    initial_vec = np.array(
        [float(p["initial_value"]) for p in param_cfgs], dtype=float
    )
    if np.any(a_vec >= b_vec):
        raise ValueError(
            "每个参数都要满足 min_value < max_value，得到 "
            + ", ".join(
                f"{p['name']}=[{format_value(a)},{format_value(b)}]"
                for p, a, b in zip(param_cfgs, a_vec, b_vec)
            )
        )
    if np.any(initial_vec < a_vec) or np.any(initial_vec > b_vec):
        raise ValueError(
            "initial_value 不在 [min_value, max_value] 内："
            + ", ".join(
                f"{p['name']}={format_value(iv)}∉[{format_value(a)},{format_value(b)}]"
                for p, iv, a, b in zip(param_cfgs, initial_vec, a_vec, b_vec)
            )
        )

    tol = float(opt_cfg.get("convergence_tolerance", 1e-3))
    max_eval = int(opt_cfg.get("max_function_evaluations", 30))
    max_relap_retries = int(opt_cfg.get("max_relap_retries", 10))
    optimizer_kind = str(opt_cfg.get("optimizer", "golden_section")).lower()
    exploration_xi = float(opt_cfg.get("exploration_xi", 0.01))
    random_seed = int(opt_cfg.get("random_seed", 12345))
    ga_crossover_probability = float(
        opt_cfg.get("ga_crossover_probability", 0.9)
    )
    ga_mutation_scale = float(opt_cfg.get("ga_mutation_scale", 0.12))

    if n_params > 1 and optimizer_kind == "golden_section":
        # 多参数下黄金分割不可用；自动切换并提示
        optimizer_kind = "bayesian"

    history: List[Dict[str, Any]] = []
    last_safe: Dict[str, Optional[np.ndarray]] = {"x": None}

    def emit(msg: str) -> None:
        if on_log is not None:
            on_log(msg)
        else:
            print(msg)

    emit(f"Environment: {config.get('env', env)}")
    emit(
        f"Objectives ({len(obj_cfgs)})："
        + " | ".join(
            f"{o.get('name','obj'+str(i))}@col{int(o['value_column_index'])}"
            f"·ref={float(o['reference_value']):g}·{o.get('deviation','absolute')}"
            f"·w={float(o.get('weight',1.0)):g}/scale={float(o.get('scale',1.0)):g}"
            for i, o in enumerate(obj_cfgs)
        )
        + f" → 加权和 f(x) = Σ w·(∫|Δ|dt)/scale"
    )
    emit(
        f"Parameters ({n_params})："
        + " | ".join(
            f"{p['name']}∈[{format_value(a)},{format_value(b)}] init={format_value(iv)}"
            for p, a, b, iv in zip(param_cfgs, a_vec, b_vec, initial_vec)
        )
    )
    emit(
        f"Optimizer = {optimizer_kind}, max evals ≈ {max_eval},"
        f" max RELAP retries per eval = {max_relap_retries}"
    )

    eval_counter = {"n": 0}

    def evaluate_record(x_request_arr: np.ndarray) -> Dict[str, Any]:
        x_request_arr = np.atleast_1d(np.asarray(x_request_arr, dtype=float))
        if stop_requested is not None and stop_requested():
            return {
                "evaluation": eval_counter["n"] + 1,
                "parameter_value": float(x_request_arr[0]),
                "parameter_values": x_request_arr.tolist(),
                "requested_parameter_values": x_request_arr.tolist(),
                "retries": 0,
                "attempted_values": [],
                "relap_failed": True,
                "relap_last_line": "stopped_by_user",
                "relap_matched_pattern": None,
                "objective_value": float("inf"),
                "objective_values": [float("inf")] * len(obj_cfgs),
            }
        eval_counter["n"] += 1
        n = eval_counter["n"]
        emit(
            f"\n--- Evaluation {n}: requested parameters = "
            f"{_vec_or_scalar_to_str(x_request_arr)} ---"
        )

        candidate, diag, attempted = _try_relap_with_recovery(
            x_request=x_request_arr,
            last_safe_x=last_safe["x"],
            initial_value=initial_vec,
            bracket=list(zip(a_vec.tolist(), b_vec.tolist())),
            max_retries=max_relap_retries,
            indta_path=indta_path,
            param_cfgs=param_cfgs,
            emit=emit,
            stop_requested=stop_requested,
        )

        # 1D 兼容字段：parameter_value（标量）取 candidate[0]，
        # 多参数下取第一个参数当摘要值。完整向量始终在 parameter_values 里。
        cand_list = candidate.tolist()
        base_rec: Dict[str, Any] = {
            "evaluation": n,
            "parameter_value": float(cand_list[0]),  # legacy / 摘要
            "parameter_values": list(cand_list),     # 多参数完整向量
            "requested_parameter_values": x_request_arr.tolist(),
            "retries": max(0, len(attempted) - 1),
            "attempted_values": [list(v.tolist()) for v in attempted],
            "relap_failed": bool(diag["failed"]),
            "relap_last_line": diag.get("last_line"),
            "relap_matched_pattern": diag.get("matched_pattern"),
        }

        if diag["failed"]:
            emit(
                f"  ✗ {max_relap_retries+1} 次尝试仍失败 / 无可回退点；"
                "将该评估目标设为 +inf。"
            )
            rec = {
                **base_rec,
                "objective_value": float("inf"),
                "objective_values": [float("inf")] * len(obj_cfgs),
            }
            history.append(rec)
            if on_iteration is not None:
                on_iteration(rec)
            return rec

        last_safe["x"] = candidate.copy()

        try:
            emit("Running plot/strip.bat → extract（plot/strip.i 列更全）...")
            meta = refresh_live_csv(
                base_dir=BASE_DIR,
                output_csv_rel=csv_rel,
            )
            if not meta.get("success"):
                raise RuntimeError(
                    "plot/strip.bat 或 extract 失败："
                    + str(meta.get("error") or "未知")
                )
            csv_path_local = str(meta["csv_path"])
            emit(
                f"  → plot/strip 完成 · {meta['rows']} 时间步 → {csv_path_local}"
            )

            if not os.path.isfile(csv_path_local):
                raise RuntimeError(f"plot/strip 完成但 CSV 缺失: {csv_path_local}")

            agg, parts = compute_weighted_objective(
                csv_path_local,
                time_col,
                obj_cfgs,
            )
        except Exception as exc:
            emit(f"  ✗ 后处理失败（plot/strip / extract / 积分）：{exc}")
            rec = {
                **base_rec,
                "objective_value": float("inf"),
                "objective_values": [float("inf")] * len(obj_cfgs),
                "post_run_error": str(exc),
            }
            history.append(rec)
            if on_iteration is not None:
                on_iteration(rec)
            return rec

        # 多目标分项打印 + 加权和
        for cfg, integ in zip(obj_cfgs, parts):
            emit(
                f"  ✓ {cfg.get('name','obj')} ∫|Δ|dt = {integ:.6g}"
                f"（col={int(cfg['value_column_index'])},"
                f" ref={float(cfg['reference_value']):g},"
                f" w={float(cfg.get('weight',1.0)):g},"
                f" scale={float(cfg.get('scale',1.0)):g}）"
            )
        emit(f"  ⇒ 加权和目标 f(x) = {agg:.6g}（{len(obj_cfgs)} 个目标加权后）")

        rec = {
            **base_rec,
            "objective_value": float(agg),
            "objective_values": [float(p) for p in parts],
        }
        history.append(rec)
        if on_iteration is not None:
            on_iteration(rec)
        return rec

    def evaluate_at(x_request_arr: np.ndarray) -> float:
        rec = evaluate_record(x_request_arr)
        try:
            return float(rec.get("objective_value", float("inf")))
        except (TypeError, ValueError):
            return float("inf")

    # === 第 1 次评估：用户配置的 initial_value 向量 ===
    emit(
        f"\n=== 第 1 次评估：initial_values = "
        f"{_vec_or_scalar_to_str(initial_vec)} ==="
    )
    initial_rec = evaluate_record(initial_vec)
    initial_obj = float(initial_rec.get("objective_value", float("inf")))
    def objective_values_from_record(rec: Dict[str, Any]) -> List[float]:
        vals = rec.get("objective_values")
        if vals is None:
            vals = [float("inf")] * len(obj_cfgs)
        try:
            out = [float(v) for v in vals]
        except (TypeError, ValueError):
            out = [float("inf")] * len(obj_cfgs)
        if len(out) != len(obj_cfgs):
            out = (out + [float("inf")] * len(obj_cfgs))[:len(obj_cfgs)]
        return out

    def find_best_history_record(
        x_best: np.ndarray, f_best_value: float
    ) -> Dict[str, Any]:
        for rec in history:
            try:
                obj = float(rec.get("objective_value", float("inf")))
            except (TypeError, ValueError):
                continue
            tol_obj = 1e-8 * (1.0 + abs(f_best_value))
            if not math.isfinite(obj) or abs(obj - f_best_value) > tol_obj:
                continue
            vals = rec.get("parameter_values")
            if vals is None and "parameter_value" in rec:
                vals = [rec["parameter_value"]]
            try:
                x_rec = np.asarray(vals, dtype=float).flatten()
            except (TypeError, ValueError):
                continue
            if x_rec.shape == x_best.shape and np.allclose(
                x_rec, x_best, rtol=1e-7, atol=1e-9
            ):
                return rec
        return initial_rec

    pareto_front: List[Dict[str, Any]] = []
    ga_best_rec: Optional[Dict[str, Any]] = None
    final_objective_values = objective_values_from_record(initial_rec)
    emit(f"  → 初值评估完成：f(initial) = {initial_obj:.6g}")

    if stop_requested is not None and stop_requested():
        x_best_vec = initial_vec.copy()
        f_best = initial_obj
        stop_reason = "stopped_by_user"
    else:
        bounds = list(zip(a_vec.tolist(), b_vec.tolist()))
        if optimizer_kind == "bayesian":
            emit(
                f"\n=== 贝叶斯优化（GP+EI · {n_params}-D · 最多 {max_eval} 次评估 ·"
                f" exploration_xi={exploration_xi:g}） ==="
            )
            warm = (
                [(initial_vec.tolist(), initial_obj)]
                if math.isfinite(initial_obj)
                else None
            )
            x_opt_vec, f_opt = bo_minimize_nd(
                evaluate_at,
                bounds,
                max_evaluations=max_eval,
                warm_start=warm,
                n_initial=max(3, n_params + 1),
                exploration_xi=exploration_xi,
            )
        elif optimizer_kind == "genetic":
            emit(
                f"\n=== Genetic optimizer (NSGA-II / Pareto) "
                f"{n_params}-D, max evaluations = {max_eval}, seed = {random_seed} ==="
            )
            ga_result = ga_minimize_pareto(
                evaluate_record,
                bounds,
                max_evaluations=max_eval,
                warm_start_records=[initial_rec],
                random_seed=random_seed,
                crossover_probability=ga_crossover_probability,
                mutation_scale=ga_mutation_scale,
                stop_requested=stop_requested,
            )
            pareto_front = list(ga_result.get("pareto_front", []))
            ga_best_rec = ga_result.get("best_record")
            if ga_best_rec:
                x_opt_vec = np.atleast_1d(
                    np.asarray(ga_best_rec["parameter_values"], dtype=float)
                )
                f_opt = float(ga_best_rec.get("objective_value", float("inf")))
            else:
                x_opt_vec = np.full(n_params, np.nan)
                f_opt = float("inf")
        elif n_params == 1:
            emit(f"\n=== 黄金分割搜索（最多 {max_eval} 次评估） ===")
            # 1D 黄金分割：把 ndarray[1] 当标量给老接口
            def _f_scalar(x: float) -> float:
                return float(evaluate_at(np.array([float(x)])))
            x_opt_scalar, f_opt = golden_section_minimize(
                _f_scalar, float(a_vec[0]), float(b_vec[0]),
                tol=tol, max_evaluations=max_eval,
            )
            x_opt_vec = np.array([float(x_opt_scalar)], dtype=float)
        else:
            raise ValueError(
                f"optimizer={optimizer_kind!r} 不支持 {n_params}-D（请使用 'bayesian'）"
            )

        # 取「初值预评估」与「优化器探索」中的全局最优
        if math.isfinite(initial_obj) and (
            not math.isfinite(f_opt) or initial_obj <= f_opt
        ):
            x_best_vec = initial_vec.copy()
            f_best = initial_obj
            final_objective_values = objective_values_from_record(initial_rec)
        else:
            x_best_vec = np.atleast_1d(np.asarray(x_opt_vec, dtype=float)).copy()
            f_best = float(f_opt)
            if optimizer_kind == "genetic" and ga_best_rec:
                final_objective_values = objective_values_from_record(ga_best_rec)
            else:
                final_objective_values = objective_values_from_record(
                    find_best_history_record(x_best_vec, f_best)
                )

        if math.isfinite(f_best):
            lines = load_indta(indta_path)
            new_lines = update_parameters_in_indta(
                lines, param_cfgs, x_best_vec.tolist()
            )
            save_indta(indta_path, new_lines)
            if optimizer_kind == "genetic" and "ga_result" in locals():
                stop_reason = str(ga_result.get("stop_reason", "optimizer_finished"))
                if stop_reason == "max_evaluations_reached":
                    stop_reason = "optimizer_finished"
            else:
                stop_reason = "optimizer_finished"
        else:
            stop_reason = "all_evaluations_failed"

    emit(
        f"\nBest parameters ≈ {_vec_or_scalar_to_str(x_best_vec)}, "
        f"weighted objective ≈ {f_best:g}"
    )
    return {
        "success": math.isfinite(f_best),
        "stop_reason": stop_reason,
        "history": history,
        # legacy 标量字段（仅参数维度=1 时有意义；多参数时取第一个分量做摘要）
        "final_parameter_value": float(x_best_vec[0]),
        # 完整向量字段（多参数推荐用这个）
        "final_parameter_values": x_best_vec.tolist(),
        "final_objective_value": f_best,
        "final_objective_values": final_objective_values,
        "pareto_front": pareto_front,
        "config": config,
    }


def run_optimization(
    env: str = "dev",
    config_override: Optional[Dict[str, Any]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    on_iteration: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Public optimization entry point.

    The legacy ``target_band`` branch has been retired; the platform now runs
    the multi-parameter, multi-objective time-integral workflow exclusively.
    """
    raw_config = config_override if config_override is not None else load_config(env)
    config = dict(raw_config)
    objective = str(config.get("objective", "minimize_time_integral")).lower()
    if objective != "minimize_time_integral":
        raise ValueError(
            "仅支持 objective='minimize_time_integral'；target_band 旧流程已移除。"
        )
    config["objective"] = "minimize_time_integral"
    return run_optimization_minimize_time_integral(
        env=env,
        config_override=config,
        stop_requested=stop_requested,
        on_log=on_log,
        on_iteration=on_iteration,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatic parameter optimization driver for RELAP5."
    )
    parser.add_argument(
        "--env",
        choices=["dev", "test", "prod"],
        default="dev",
        help="Environment configuration to use (default: dev).",
    )
    args = parser.parse_args()
    run_optimization(env=args.env)


if __name__ == "__main__":
    main()
