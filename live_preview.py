"""
RELAP5 优化系统 — 实时预览模块。

在 RELAP 计算过程中（每次 evaluation 往往要数分钟），从最新 ``rstplt`` 增量
生成 ``plot/stripf`` 与 CSV，供 UI 直接画 ``p(t)`` —— 无需等待整次评估完成。

复用 ``extract_plotrec_rows`` 的 **Python API**（不再 subprocess 调它的 CLI）。

两条路径**共用同一份输出 CSV** ``plot/stripf_plotrec_row.csv``，通过模块级
:data:`CSV_WRITE_LOCK` 串行写入避免内容交错：

================  =======================================  =========================================
角色              主流程（auto_optimize.evaluate_at）        实时预览（refresh_live_csv）
================  =======================================  =========================================
strip.bat         BASE_DIR/strip.bat（cwd=BASE_DIR）         BASE_DIR/plot/strip.bat（cwd=plot/）
stripf 中间产物    BASE_DIR/stripf                            BASE_DIR/plot/stripf
**输出 CSV**       共用 BASE_DIR/plot/stripf_plotrec_row.csv（互斥锁串行）
================  =======================================  =========================================

只读共享 ``BASE_DIR/rstplt``（``shutil.copy2`` 走 Windows 默认共享读，安全）。
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# extract_plotrec_rows.py 现仅存在于 plot/ 子目录；用 importlib 做路径无关加载，
# 顺序与 auto_optimize._extract_plotrec_csv 的候选路径一致。
_THIS_DIR = Path(__file__).resolve().parent
_EXTRACT_CANDIDATES = (
    _THIS_DIR / "extract_plotrec_rows.py",
    _THIS_DIR / "plot" / "extract_plotrec_rows.py",
)


def _load_extract_module() -> Any:
    for p in _EXTRACT_CANDIDATES:
        if p.is_file():
            spec = importlib.util.spec_from_file_location(
                "extract_plotrec_rows", str(p)
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules.setdefault("extract_plotrec_rows", mod)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                return mod
    raise ImportError(
        "extract_plotrec_rows.py 未在以下任一位置找到：\n  "
        + "\n  ".join(str(p) for p in _EXTRACT_CANDIDATES)
    )


_extract = _load_extract_module()
build_csv_header = _extract.build_csv_header
parse_plotrec_rows = _extract.parse_plotrec_rows
parse_stripf_headers = _extract.parse_stripf_headers
write_rows_to_csv = _extract.write_rows_to_csv


# ---- 共享写锁 ----------------------------------------------------------------
# 所有写/拷贝 ``plot/stripf_plotrec_row.csv`` 的代码都必须 acquire 这个锁，
# 防止「优化主流程」（auto_optimize.extract_plotrec_csv）与「实时预览线程」
# （refresh_live_csv）在同一时刻覆盖同一份文件造成内容交错。
#
# 使用方式：
#     from live_preview import CSV_WRITE_LOCK
#     with CSV_WRITE_LOCK:
#         ...  # 写或拷贝 live.csv
CSV_WRITE_LOCK = threading.Lock()


def refresh_live_csv(
    base_dir: str,
    output_csv_rel: str = "plot/stripf_plotrec_row.csv",
    rstplt_root_rel: str = "rstplt",
    plot_subdir: str = "plot",
    strip_bat_name: str = "strip.bat",
    stripf_name: str = "stripf",
    strip_timeout_sec: float = 60.0,
    skip_if_rstplt_unchanged_since: Optional[float] = None,
) -> Dict[str, Any]:
    """
    单次刷新流程：

    ``base_dir/rstplt`` → 复制到 ``plot/rstplt`` → 运行 ``plot/strip.bat`` →
    解析 ``plot/stripf``（``extract_plotrec_rows``） → 写入 ``base_dir/<output_csv_rel>``。

    Returns
    -------
    dict
        ``{success, error, rows, csv_path, updated_at, rstplt_mtime}``。
        永远返回字典，**不抛异常**给调用方（适合后台线程消费）。

    Notes
    -----
    - 当 ``skip_if_rstplt_unchanged_since`` 给定且当前 rstplt mtime 不大于该值时，
      不做任何 IO，直接返回 ``success=False, error=None``（"无更新可静默跳过"语义）。
    - 写到与实时预览图一致的 plotrec CSV 路径，并通过锁与优化器主流程串行。
    """
    base = Path(base_dir)
    rstplt_root = base / rstplt_root_rel
    plot_dir = base / plot_subdir
    rstplt_plot = plot_dir / "rstplt"
    strip_bat = plot_dir / strip_bat_name
    stripf_plot = plot_dir / stripf_name
    csv_out = base / output_csv_rel

    out: Dict[str, Any] = {
        "success": False,
        "error": None,
        "rows": 0,
        "csv_path": str(csv_out),
        "updated_at": time.time(),
        "rstplt_mtime": 0.0,
    }

    if not rstplt_root.is_file():
        out["error"] = f"rstplt 不存在：{rstplt_root}"
        return out

    try:
        m = rstplt_root.stat().st_mtime
        out["rstplt_mtime"] = m
    except OSError as exc:
        out["error"] = f"读取 rstplt 状态失败：{exc}"
        return out

    if skip_if_rstplt_unchanged_since is not None and m <= float(
        skip_if_rstplt_unchanged_since
    ):
        return out

    if not strip_bat.is_file():
        out["error"] = f"plot/strip.bat 不存在：{strip_bat}"
        return out

    try:
        plot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rstplt_root, rstplt_plot)
    except Exception as exc:
        out["error"] = f"复制 rstplt 失败：{exc}"
        return out

    try:
        proc = subprocess.run(
            ["cmd", "/c", str(strip_bat)],
            cwd=str(plot_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=strip_timeout_sec,
        )
    except subprocess.TimeoutExpired:
        out["error"] = "plot/strip.bat 超时"
        return out
    except Exception as exc:
        out["error"] = f"plot/strip.bat 启动失败：{exc}"
        return out

    if proc.returncode != 0:
        out["error"] = f"plot/strip.bat 失败 exit={proc.returncode}"
        return out
    if not stripf_plot.is_file():
        out["error"] = f"strip 后未生成 stripf：{stripf_plot}"
        return out

    try:
        names, nums = parse_stripf_headers(stripf_plot)
        rows = parse_plotrec_rows(stripf_plot)
        if not rows:
            out["error"] = "未解析出 plotrec 记录"
            return out
        max_cols = max(len(r) for r in rows)
        header = build_csv_header(names, nums, max_cols)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with CSV_WRITE_LOCK:
            write_rows_to_csv(rows, csv_out, header)
    except Exception as exc:
        out["error"] = f"解析或写 CSV 失败：{exc}"
        return out

    out["success"] = True
    out["rows"] = len(rows)
    out["updated_at"] = time.time()
    return out


def start_live_preview_loop(
    stop_event: threading.Event,
    base_dir: str,
    output_csv_rel: str = "plot/stripf_plotrec_row.csv",
    interval_sec: float = 20.0,
    on_log: Optional[Callable[[str], None]] = None,
    on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    grace_period_sec: float = 5.0,
) -> threading.Thread:
    """
    以守护线程方式每 ``interval_sec`` 调用一次 :func:`refresh_live_csv`，
    仅当 rstplt 有更新时才真正写盘并触发 ``on_update``。

    通过 ``stop_event.set()`` 即可让线程退出（最多在下一次 wait 后立即结束）。
    永不向调用方抛异常。
    """
    last_mtime: Dict[str, float] = {"v": 0.0}

    def _safe_call(fn: Optional[Callable[..., Any]], *args: Any) -> None:
        if fn is None:
            return
        try:
            fn(*args)
        except Exception:
            pass

    def _loop() -> None:
        if grace_period_sec > 0:
            stop_event.wait(timeout=grace_period_sec)
        while not stop_event.is_set():
            try:
                meta = refresh_live_csv(
                    base_dir=base_dir,
                    output_csv_rel=output_csv_rel,
                    skip_if_rstplt_unchanged_since=last_mtime["v"],
                )
                if meta.get("success"):
                    last_mtime["v"] = float(meta.get("rstplt_mtime", 0.0))
                    _safe_call(on_update, meta)
                    _safe_call(
                        on_log,
                        f"[实时预览] 已刷新 · {meta['rows']} 行 → {meta['csv_path']}",
                    )
                elif meta.get("error"):
                    _safe_call(on_log, f"[实时预览][WARN] {meta['error']}")
            except Exception as exc:
                _safe_call(on_log, f"[实时预览][ERROR] 未知异常：{exc}")
            stop_event.wait(timeout=max(0.5, float(interval_sec)))

    t = threading.Thread(target=_loop, daemon=True, name="live-preview-loop")
    t.start()
    return t
