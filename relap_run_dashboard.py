import locale
import csv
import queue
import shutil
import subprocess
import threading
import time
import math
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from extract_plotrec_rows import (
    build_csv_header,
    parse_plotrec_rows,
    parse_stripf_headers,
    write_rows_to_csv,
)


BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR / "plot"
START_BAT = BASE_DIR / "start.bat"
STRIP_BAT = PLOT_DIR / "strip.bat"
EXTRACT_SCRIPT = PLOT_DIR / "extract_plotrec_rows.py"
STRIPF_PATH = PLOT_DIR / "stripf"
RSTPLT_PATH = BASE_DIR / "rstplt"
PLOT_RSTPLT_PATH = PLOT_DIR / "rstplt"
RSTPLT_R_PATH = PLOT_DIR / "rstplt.r"
CSV_PATH = PLOT_DIR / "stripf_plotrec_rows.csv"
SCREEN_PATH = PLOT_DIR / "screen"
OUTDTA_PATH = PLOT_DIR / "outdta"
# Live preview: sync rstplt and run strip; keep interval small enough to track run.
LIVE_UPDATE_INTERVAL_SEC = 20.0
CONSOLE_READER_BLOCK = 4096
ENC = locale.getpreferredencoding(False) or "utf-8"
STRIP_WORKFLOW_LOCK = threading.Lock()


def _pipe_reader_to_queue(
    stream: object,
    log_q: "queue.Queue[str]",
) -> None:
    """Read child stdout and emit updates split by CR/LF."""
    if stream is None:
        return
    text_buffer = ""
    while True:
        chunk = stream.read(CONSOLE_READER_BLOCK)
        if not chunk:
            break
        try:
            piece = chunk.decode(ENC, errors="replace")
        except Exception:
            piece = str(chunk)
        # RELAP often refreshes status using '\r' in-place updates.
        text_buffer += piece.replace("\r", "\n")
        while "\n" in text_buffer:
            line, text_buffer = text_buffer.split("\n", 1)
            msg = line.strip()
            if msg:
                log_q.put(msg)
    if text_buffer.strip():
        try:
            text = text_buffer.strip()
        except Exception:
            text = str(text_buffer)
        if text.strip():
            log_q.put(text.strip())
    # Do not call stream.close(); Popen owns the handle.


def _tail_file_increment(
    file_path: Path,
    pos_holder: Dict[str, int],
    log_q: "queue.Queue[str]",
    prefix: str,
) -> None:
    if not file_path.exists():
        return
    pos = pos_holder.get(str(file_path), 0)
    try:
        with open(file_path, "r", encoding=ENC, errors="replace") as f:
            f.seek(pos)
            chunk = f.read()
            pos_holder[str(file_path)] = f.tell()
    except (OSError, PermissionError):
        return
    if not chunk:
        return
    for raw in chunk.splitlines():
        line = raw.strip()
        if line:
            log_q.put(f"[{prefix}] {line}")


def run_bat_with_logs(
    bat_path: Path,
    log_q: "queue.Queue[str]",
    on_progress: object = None,
    fallback_tail_files: Optional[List[Path]] = None,
    working_dir: Optional[Path] = None,
) -> None:
    if not bat_path.exists():
        raise FileNotFoundError(f"Batch file not found: {bat_path}")

    # Use PIPE + a reader thread. Writing to a log file and tailing the same file from
    # the parent does not work on Windows (second open often PermissionError) so nothing
    # appeared in the UI during the run.
    bat_abs = bat_path.resolve()
    log_q.put(f"[INFO] Running: {bat_path.name}")

    proc = subprocess.Popen(
        ["cmd.exe", "/c", str(bat_abs)],
        cwd=str(working_dir or BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    reader = threading.Thread(
        target=_pipe_reader_to_queue,
        args=(proc.stdout, log_q),
        daemon=True,
    )
    reader.start()

    start_ts = time.monotonic()
    next_heartbeat = start_ts + 2.0
    file_positions: Dict[str, int] = {}
    tail_files = fallback_tail_files or []
    while proc.poll() is None:
        now = time.monotonic()
        elapsed = now - start_ts
        for p in tail_files:
            prefix = "screen" if p == SCREEN_PATH else "outdta"
            _tail_file_increment(p, file_positions, log_q, prefix=prefix)
        if on_progress is not None:
            try:
                on_progress(elapsed)
            except Exception as progress_exc:
                log_q.put(f"[WARN] progress callback error: {progress_exc}")
        if now >= next_heartbeat:
            log_q.put(f"[INFO] {bat_path.name} running... {int(elapsed)}s")
            next_heartbeat = now + 2.0
        time.sleep(0.2)

    ret = proc.wait()
    reader.join(timeout=5.0)
    for p in tail_files:
        prefix = "screen" if p == SCREEN_PATH else "outdta"
        _tail_file_increment(p, file_positions, log_q, prefix=prefix)

    if ret != 0:
        raise RuntimeError(f"{bat_path.name} failed with exit code {ret}")
    log_q.put(f"[OK] {bat_path.name} finished")


def parse_stripf_to_chart_data(
    stripf_path: Path, csv_path: Path
) -> Dict[str, object]:
    if not stripf_path.exists():
        raise FileNotFoundError(f"stripf not found: {stripf_path}")

    names, nums = parse_stripf_headers(stripf_path)
    records = parse_plotrec_rows(stripf_path)
    if not records:
        raise RuntimeError("No plotrec records found in stripf.")

    max_cols = max(len(r) for r in records)
    header = build_csv_header(names, nums, max_cols)
    write_rows_to_csv(records, csv_path, header)

    chart_rows: List[Dict[str, float]] = []
    x_name = header[0] if header else "time"
    y_cols = header[1:] if len(header) > 1 else []

    for rec in records:
        row: Dict[str, float] = {}
        for idx, col_name in enumerate(header):
            row[col_name] = rec[idx] if idx < len(rec) else None
        chart_rows.append(row)

    return {
        "rows_count": len(records),
        "header": header,
        "chart_rows": chart_rows,
        "x_name": x_name,
        "y_cols": y_cols,
        "csv_path": str(csv_path),
    }


def run_extract_plotrec_rows(log_q: "queue.Queue[str]") -> None:
    if not EXTRACT_SCRIPT.exists():
        raise FileNotFoundError(f"extract script not found: {EXTRACT_SCRIPT}")
    cmd = [
        "python",
        str(EXTRACT_SCRIPT),
        "--input",
        str(STRIPF_PATH),
        "--output",
        str(CSV_PATH),
    ]
    attempts = 3
    last_stdout = ""
    for i in range(1, attempts + 1):
        proc = subprocess.run(
            cmd,
            cwd=str(PLOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        last_stdout = proc.stdout or ""
        if last_stdout:
            for line in last_stdout.splitlines():
                msg = line.strip()
                if msg:
                    log_q.put(f"[extract] {msg}")
        if proc.returncode == 0:
            return
        log_q.put(
            f"[WARN] extract attempt {i}/{attempts} failed, exit={proc.returncode}"
        )
        time.sleep(0.8)

    tail = ""
    if last_stdout:
        lines = [ln.strip() for ln in last_stdout.splitlines() if ln.strip()]
        tail = " | ".join(lines[-6:])
    raise RuntimeError(
        "extract_plotrec_rows.py failed after retries."
        + (f" details: {tail}" if tail else "")
    )


def load_chart_data_from_csv(csv_path: Path) -> Dict[str, object]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    chart_rows: List[Dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for raw in reader:
            row: Dict[str, float] = {}
            for col in header:
                value = raw.get(col, "")
                if value in ("", None):
                    row[col] = None
                else:
                    row[col] = float(value)
            chart_rows.append(row)
    if not chart_rows:
        raise RuntimeError("No rows loaded from extracted CSV.")
    x_name = header[0] if header else "time"
    y_cols = header[1:] if len(header) > 1 else []
    return {
        "rows_count": len(chart_rows),
        "header": header,
        "chart_rows": chart_rows,
        "x_name": x_name,
        "y_cols": y_cols,
        "csv_path": str(csv_path),
    }


def refresh_live_preview(
    log_q: "queue.Queue[str]",
    preview_q: "queue.Queue[Dict[str, object]]",
    min_rstplt_mtime: Optional[float] = None,
) -> None:
    if not RSTPLT_PATH.exists():
        return
    try:
        rstplt_mtime = RSTPLT_PATH.stat().st_mtime
    except OSError:
        return
    # Guard against using stale rstplt from a previous completed run.
    if min_rstplt_mtime is not None and rstplt_mtime < min_rstplt_mtime:
        return

    with STRIP_WORKFLOW_LOCK:
        try:
            # plot/strip.bat expects plot/rstplt, then copies it to plot/rstplt.r
            shutil.copy2(RSTPLT_PATH, PLOT_RSTPLT_PATH)
        except Exception as copy_exc:
            log_q.put(f"[WARN] test/rstplt -> plot/rstplt sync failed: {copy_exc}")
            return

        proc = subprocess.run(
            ["cmd", "/c", str(STRIP_BAT)],
            cwd=str(PLOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if proc.returncode != 0:
            log_q.put(f"[WARN] live strip failed, exit={proc.returncode}")
            return

        try:
            parse_result = parse_stripf_to_chart_data(STRIPF_PATH, CSV_PATH)
            parse_result["success"] = True
            parse_result["mode"] = "live"
            preview_q.put(parse_result)
            log_q.put(
                f"[LIVE] Updated preview from rstplt, timesteps={parse_result['rows_count']}"
            )
        except Exception as parse_exc:
            log_q.put(f"[WARN] live parse failed: {parse_exc}")


def drain_queues() -> None:
    log_q: "queue.Queue[str]" = st.session_state.log_q
    result_q: "queue.Queue[Dict[str, object]]" = st.session_state.result_q
    preview_q: "queue.Queue[Dict[str, object]]" = st.session_state.preview_q

    while True:
        try:
            st.session_state.logs.append(log_q.get_nowait())
        except queue.Empty:
            break

    while True:
        try:
            result = result_q.get_nowait()
            st.session_state.last_result = result
            mode = result.get("mode")
            if mode == "relap":
                st.session_state.running = False
            elif mode == "refresh":
                st.session_state.refresh_running = False
                if result.get("success") and result.get("chart_rows"):
                    st.session_state.live_result = result
        except queue.Empty:
            break

    while True:
        try:
            st.session_state.live_result = preview_q.get_nowait()
        except queue.Empty:
            break


def start_worker_run_relap() -> None:
    """Run start.bat with live log; optional live chart from rstplt during run."""
    st.session_state.running = True
    st.session_state.refresh_running = False
    st.session_state.logs = []
    st.session_state.last_result = None
    st.session_state.log_q = queue.Queue()
    st.session_state.result_q = queue.Queue()
    st.session_state.preview_q = queue.Queue()
    st.session_state.live_result = None
    log_q: "queue.Queue[str]" = st.session_state.log_q
    result_q: "queue.Queue[Dict[str, object]]" = st.session_state.result_q
    preview_q: "queue.Queue[Dict[str, object]]" = st.session_state.preview_q

    def task() -> None:
        result: Dict[str, object] = {"success": False, "mode": "relap"}
        try:
            run_started_at = time.time()
            last_live_update_elapsed = -LIVE_UPDATE_INTERVAL_SEC

            def maybe_live_update(elapsed: float) -> None:
                nonlocal last_live_update_elapsed
                if elapsed - last_live_update_elapsed < LIVE_UPDATE_INTERVAL_SEC:
                    return
                refresh_live_preview(
                    log_q,
                    preview_q,
                    min_rstplt_mtime=run_started_at,
                )
                last_live_update_elapsed = elapsed

            run_bat_with_logs(
                START_BAT,
                log_q,
                on_progress=maybe_live_update,
            )
            refresh_live_preview(
                log_q,
                preview_q,
                min_rstplt_mtime=run_started_at,
            )
            # 将当前 stripf 解析进结果，避免界面仍显示上一次「刷新画图」的曲线
            if STRIPF_PATH.exists():
                try:
                    final_charts = parse_stripf_to_chart_data(
                        STRIPF_PATH, CSV_PATH
                    )
                    result.update(final_charts)
                    result["mode"] = "relap"
                    log_q.put(
                        f"[OK] 已根据当前 stripf 更新曲线数据：{final_charts['rows_count']} 个时间步"
                    )
                except Exception as pe:
                    log_q.put(
                        f"[WARN] 运行结束但未能解析当前 stripf（可再点「刷新画图」）: {pe}"
                    )
            result["success"] = True
        except Exception as exc:
            log_q.put(f"[ERROR] {exc}")
            result["error"] = str(exc)

        result_q.put(result)

    st.session_state.worker_thread = threading.Thread(target=task, daemon=True)
    st.session_state.worker_thread.start()


def start_worker_refresh_plots() -> None:
    """rstplt -> rstplt.r, run strip.bat, re-parse stripf for charts (append to log)."""
    st.session_state.refresh_running = True
    log_q: "queue.Queue[str]" = st.session_state.log_q
    result_q: "queue.Queue[Dict[str, object]]" = st.session_state.result_q
    min_rstplt_mtime = st.session_state.get("relap_run_started_at")

    def task() -> None:
        result: Dict[str, object] = {"success": False, "mode": "refresh"}
        try:
            log_q.put("[INFO] --- 刷新画图流程开始 ---")
            with STRIP_WORKFLOW_LOCK:
                if RSTPLT_PATH.exists():
                    if min_rstplt_mtime is not None:
                        try:
                            rstplt_mtime = RSTPLT_PATH.stat().st_mtime
                        except OSError:
                            rstplt_mtime = 0.0
                        if rstplt_mtime < float(min_rstplt_mtime):
                            log_q.put(
                                "[WARN] 当前 rstplt 可能还是上一次运行结果，将继续刷新并使用当前可用数据。"
                            )
                    shutil.copy2(RSTPLT_PATH, PLOT_RSTPLT_PATH)
                    log_q.put("[OK] 已复制 test/rstplt -> plot/rstplt")
                else:
                    log_q.put("[WARN] 未找到 rstplt，仍将尝试运行 strip.bat")

                log_q.put("[INFO] 步骤1/3：运行 strip.bat 生成 stripf")
                run_bat_with_logs(STRIP_BAT, log_q, working_dir=PLOT_DIR)

                log_q.put("[INFO] 步骤2/3：运行 extract_plotrec_rows.py 提取 stripf")
                use_csv = True
                try:
                    run_extract_plotrec_rows(log_q)
                except Exception as extract_exc:
                    # Do not fail whole refresh if extractor fails transiently.
                    log_q.put(f"[WARN] extract failed, fallback to direct stripf parse: {extract_exc}")
                    use_csv = False

                log_q.put("[INFO] 步骤3/3：读取提取结果并更新图表")
                if use_csv:
                    parse_result = load_chart_data_from_csv(CSV_PATH)
                else:
                    parse_result = parse_stripf_to_chart_data(STRIPF_PATH, CSV_PATH)
                log_q.put(f"[OK] 已加载曲线数据：{parse_result['rows_count']} 个时间步")
                log_q.put(f"[OK] 结果CSV：{parse_result['csv_path']}")
                result.update(parse_result)
                result["success"] = True
        except Exception as exc:
            log_q.put(f"[ERROR] {exc}")
            result["error"] = str(exc)

        result_q.put(result)

    st.session_state.worker_thread = threading.Thread(target=task, daemon=True)
    st.session_state.worker_thread.start()


def build_sampled_rows(
    rows: List[Dict[str, float]],
    start_idx: int,
    end_idx: int,
    max_points: int,
) -> List[Dict[str, float]]:
    if not rows:
        return []

    start_idx = max(0, min(start_idx, len(rows) - 1))
    end_idx = max(start_idx, min(end_idx, len(rows) - 1))
    subset = rows[start_idx : end_idx + 1]
    if not subset:
        return []

    if max_points <= 0 or len(subset) <= max_points:
        return subset

    step = math.ceil(len(subset) / max_points)
    return subset[::step]


st.set_page_config(page_title="RELAP5 运行与绘图", layout="wide")
st.title("RELAP5 运行与绘图")

for key, default in {
    "running": False,
    "refresh_running": False,
    "logs": [],
    "last_result": None,
    "worker_thread": None,
    "log_q": queue.Queue(),
    "result_q": queue.Queue(),
    "preview_q": queue.Queue(),
    "live_result": None,
    "relap_run_started_at": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

drain_queues()

col_run, col_refresh = st.columns(2)
with col_run:
    if st.button("运行 RELAP5（start.bat）", disabled=st.session_state.running, type="primary"):
        st.session_state.relap_run_started_at = time.time()
        start_worker_run_relap()
with col_refresh:
    if st.button("刷新画图", disabled=st.session_state.refresh_running):
        start_worker_refresh_plots()

st.caption(
    "运行：执行 start.bat 并在下方日志实时显示；运行中可按间隔从 rstplt 同步预览曲线。"
    " 刷新画图：将 rstplt 复制为 rstplt.r 后执行 strip.bat，再根据 stripf 重绘并导出 CSV。"
)

st.subheader("状态")
busy = st.session_state.running
refresh_busy = st.session_state.refresh_running
if busy and refresh_busy:
    st.write("RELAP运行中（同时正在刷新画图）")
elif busy:
    st.write("RELAP运行中…")
elif refresh_busy:
    st.write("正在刷新画图…")
else:
    st.write("空闲")

result = st.session_state.last_result
if result and (not busy or result.get("mode") == "refresh"):
    if result.get("success"):
        if result.get("mode") == "refresh" and result.get("rows_count") is not None:
            st.success(f"刷新画图完成：{result.get('rows_count')} 个时间步")
        elif result.get("mode") == "relap":
            n = result.get("rows_count")
            st.success(
                f"RELAP5 运行完成"
                + (f"，已载入 {n} 个时间步曲线" if n is not None else "")
            )
    elif result.get("error"):
        st.error(f"失败：{result.get('error')}")

st.subheader("运行日志")
# st.text_area 的 value 在多次 rerun 时会被会话状态“粘住”，运行中看起来不更新；用 code+容器高度代替
log_body = "\n".join(st.session_state.logs[-500:]) or "(无)"
st.code(log_body, language="text", line_numbers=False)

st.subheader("stripf 时序图")
chart_source = None
# 运行中优先实时预览；结束后有「带(chart_rows)的 last_result」优先（含刚跑完的 RELAP 解析）
if busy and st.session_state.live_result and st.session_state.live_result.get(
    "chart_rows"
):
    chart_source = st.session_state.live_result
elif result and result.get("success") and result.get("chart_rows"):
    chart_source = result
elif st.session_state.live_result and st.session_state.live_result.get("chart_rows"):
    chart_source = st.session_state.live_result

if chart_source:
    x_name = str(chart_source["x_name"])
    y_cols_all = list(chart_source["y_cols"])
    total_points = int(chart_source.get("rows_count", len(chart_source["chart_rows"])))
    default_cols = y_cols_all[: min(2, len(y_cols_all))]
    selected_cols = st.multiselect(
        "选择显示列",
        options=y_cols_all,
        default=default_cols,
    )
    range_values = st.slider(
        "时间步范围（索引）",
        min_value=0,
        max_value=max(total_points - 1, 0),
        value=(0, max(total_points - 1, 0)),
    )
    max_points = st.slider(
        "最大绘图点数（降采样）",
        min_value=200,
        max_value=5000,
        value=min(max(total_points, 200), 1200),
        step=100,
    )
    if selected_cols:
        sampled_rows = build_sampled_rows(
            chart_source["chart_rows"],
            start_idx=range_values[0],
            end_idx=range_values[1],
            max_points=max_points,
        )
        columns_per_row = 3
        for row_start in range(0, len(selected_cols), columns_per_row):
            row_cols = selected_cols[row_start : row_start + columns_per_row]
            ui_cols = st.columns(columns_per_row)
            for idx, col_name in enumerate(row_cols):
                with ui_cols[idx]:
                    st.markdown(f"**{col_name}**")
                    st.line_chart(sampled_rows, x=x_name, y=[col_name])
        st.caption(
            f"绘图点数：{len(sampled_rows)} / 原始区间点数：{range_values[1] - range_values[0] + 1}"
        )
    st.caption(f"X 轴：{x_name}")
    if chart_source.get("mode") == "live":
        st.caption("当前显示：运行中实时预览（rstplt -> rstplt.r -> strip）")
else:
    st.info("暂无可视化数据。请先运行 RELAP5 或点击「刷新画图」。")

if st.session_state.running or st.session_state.refresh_running:
    time.sleep(1)
    st.rerun()
