"""单元测试：live_preview（不依赖 RELAP / Streamlit / 真实 strip.bat）。"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest

from live_preview import (
    CSV_WRITE_LOCK,
    refresh_live_csv,
    start_live_preview_loop,
)
from optimization_trace_view import build_live_pt_figure


def _touch_binary(path: str, size: int = 16) -> None:
    with open(path, "wb") as f:
        f.write(b"\x00" * size)


class TestRefreshLiveCsvErrors(unittest.TestCase):
    """`refresh_live_csv` 永不抛异常；不同失败路径都返回明确 error 字段。"""

    def test_missing_rstplt_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = refresh_live_csv(base_dir=tmp)
            self.assertFalse(res["success"])
            self.assertIn("rstplt", str(res["error"]))

    def test_missing_strip_bat_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch_binary(os.path.join(tmp, "rstplt"))
            os.mkdir(os.path.join(tmp, "plot"))
            res = refresh_live_csv(base_dir=tmp)
            self.assertFalse(res["success"])
            self.assertIn("strip.bat", str(res["error"]))

    def test_skip_when_unchanged(self) -> None:
        """rstplt mtime 未超出 ``skip_if_rstplt_unchanged_since`` 时静默跳过。"""
        with tempfile.TemporaryDirectory() as tmp:
            rp = os.path.join(tmp, "rstplt")
            _touch_binary(rp)
            mtime = os.stat(rp).st_mtime

            res = refresh_live_csv(
                base_dir=tmp,
                skip_if_rstplt_unchanged_since=mtime + 1.0,
            )
            self.assertFalse(res["success"])
            self.assertIsNone(res["error"])
            self.assertEqual(res["rows"], 0)


class TestLivePreviewLoop(unittest.TestCase):
    def test_loop_starts_and_stops_cleanly(self) -> None:
        stop = threading.Event()
        with tempfile.TemporaryDirectory() as tmp:
            t = start_live_preview_loop(
                stop_event=stop,
                base_dir=tmp,
                interval_sec=0.05,
                grace_period_sec=0.0,
            )
            time.sleep(0.2)
            stop.set()
            t.join(timeout=2.0)
            self.assertFalse(t.is_alive())

    def test_loop_emits_logs_for_missing_rstplt(self) -> None:
        stop = threading.Event()
        captured = []
        with tempfile.TemporaryDirectory() as tmp:
            t = start_live_preview_loop(
                stop_event=stop,
                base_dir=tmp,
                interval_sec=0.05,
                grace_period_sec=0.0,
                on_log=captured.append,
            )
            time.sleep(0.25)
            stop.set()
            t.join(timeout=2.0)
        self.assertTrue(any("rstplt" in s for s in captured),
                        f"expected rstplt warning in logs; got {captured!r}")


class TestLivePtFigure(unittest.TestCase):
    """`build_live_pt_figure` 与 CSV 路径联动；缺失时返回占位空图。"""

    def test_missing_csv_returns_placeholder_figure(self) -> None:
        fig = build_live_pt_figure(
            csv_path="/no/such/path.csv",
            time_col=0, value_col=1, reference_value=0.0,
        )
        self.assertEqual(len(fig.data), 0)
        ann_text = " ".join(a.text for a in (fig.layout.annotations or []))
        self.assertIn("等待", ann_text)

    def test_with_csv_renders_traces_and_ref_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "live.csv")
            with open(p, "w", encoding="utf-8") as f:
                f.write("t,v\n0,0\n1,2\n2,4\n")
            fig = build_live_pt_figure(
                csv_path=p, time_col=0, value_col=1,
                reference_value=2.0, last_updated_at=time.time(),
            )
            self.assertGreaterEqual(len(fig.data), 4)


class TestSharedCsvWriteLock(unittest.TestCase):
    """``CSV_WRITE_LOCK`` 是优化主流程与实时预览共用的写互斥锁。"""

    def test_lock_is_threading_lock(self) -> None:
        # threading.Lock() 创建的对象是 _thread.lock 类型；用 acquire/release 验证语义
        self.assertTrue(hasattr(CSV_WRITE_LOCK, "acquire"))
        self.assertTrue(hasattr(CSV_WRITE_LOCK, "release"))

    def test_lock_is_reentrant_safe_for_with_block(self) -> None:
        # 不要求是 RLock，但 with 块必须能正常 acquire/release（不被前测留下的状态污染）
        with CSV_WRITE_LOCK:
            self.assertFalse(CSV_WRITE_LOCK.acquire(blocking=False))
        # 离开 with 后必须可重新获取
        self.assertTrue(CSV_WRITE_LOCK.acquire(blocking=False))
        CSV_WRITE_LOCK.release()


class TestUnifiedCsvDefaults(unittest.TestCase):
    """优化主流程默认 CSV 路径必须与实时预览图统一。"""

    def test_extract_plotrec_csv_default_targets_live_csv(self) -> None:
        import inspect
        from auto_optimize import extract_plotrec_csv

        sig = inspect.signature(extract_plotrec_csv)
        self.assertEqual(
            sig.parameters["csv_rel"].default,
            "plot/stripf_plotrec_row.csv",
        )

    def test_dashboard_default_objective_uses_live_csv(self) -> None:
        # 不直接 import optimization_dashboard（streamlit 副作用）；只读 JSON 默认值
        import json
        import os

        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config_dev.json",
        )
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(
            cfg["integral_objective"]["csv_relative_path"],
            "plot/stripf_plotrec_row.csv",
        )


if __name__ == "__main__":
    unittest.main()
