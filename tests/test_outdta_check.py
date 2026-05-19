"""单元测试：outdta_check（不依赖 RELAP）。"""

from __future__ import annotations

import os
import tempfile
import unittest

from outdta_check import (
    OUTDTA_FAILURE_PATTERNS,
    OUTDTA_FAILURE_PREFIXES,
    OUTDTA_SUCCESS_PATTERNS,
    outdta_failure_diagnosis,
    read_last_nonempty_line,
    read_tail_lines,
)


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


class TestReadLastNonempty(unittest.TestCase):
    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(read_last_nonempty_line("/no/such/path.o"), "")

    def test_empty_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.o")
            open(p, "w").close()
            self.assertEqual(read_last_nonempty_line(p), "")

    def test_skips_trailing_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.o")
            _write_text(p, "first\nlast meaningful line\n\n   \n\n")
            self.assertEqual(read_last_nonempty_line(p), "last meaningful line")

    def test_handles_crlf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.o")
            _write_text(p, "a\r\nb\r\nfinal line\r\n")
            self.assertEqual(read_last_nonempty_line(p), "final line")

    def test_only_reads_tail(self) -> None:
        """大文件下，仅读 max_bytes 字节也能拿到最末行。"""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "big.o")
            with open(p, "w", encoding="utf-8") as f:
                f.write("filler line\n" * 5000)
                f.write("0******** Transient terminated by failure.\n")
            self.assertEqual(
                read_last_nonempty_line(p, max_bytes=512),
                "0******** Transient terminated by failure.",
            )


class TestReadTailLines(unittest.TestCase):
    def test_returns_at_most_max_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.o")
            _write_text(p, "\n".join(f"l{i}" for i in range(1, 21)) + "\n")
            tail = read_tail_lines(p, max_lines=5)
            self.assertEqual(tail, ["l16", "l17", "l18", "l19", "l20"])


class TestDiagnosis(unittest.TestCase):
    def test_missing_file(self) -> None:
        diag = outdta_failure_diagnosis("/no/such/path.o")
        self.assertFalse(diag["exists"])
        self.assertFalse(diag["failed"])
        self.assertIsNone(diag["matched_pattern"])

    def test_failure_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.o")
            _write_text(
                p,
                "0Final time=    2.29647     sec\n"
                "0******** Transient terminated by failure.\n",
            )
            diag = outdta_failure_diagnosis(p)
            self.assertTrue(diag["exists"])
            self.assertTrue(diag["failed"])
            self.assertEqual(diag["matched_pattern"], "Transient terminated by failure")
            self.assertIn("Transient terminated by failure", diag["last_line"])

    def test_normal_termination_is_not_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.o")
            _write_text(
                p,
                "0Final time=  100.0  sec\n"
                "0Transient terminated by end of time step cards.\n",
            )
            diag = outdta_failure_diagnosis(p)
            self.assertTrue(diag["exists"])
            self.assertFalse(diag["failed"])
            self.assertIsNone(diag["matched_pattern"])

    def test_runtime_abort_without_terminal_summary_is_failure(self) -> None:
        """A partial outdta tail without a RELAP terminal marker is unsafe."""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.o")
            _write_text(
                p,
                "0terms computed for stacc\n"
                "    qwg= -1.23120E-13 amasso=   653.55\n"
                "    uliq=  2.04685E+05   p(i)=  4.13686E+06\n",
            )
            diag = outdta_failure_diagnosis(p)
            self.assertTrue(diag["failed"])
            self.assertEqual(diag["matched_pattern"], "missing_normal_termination")

    def test_case_insensitive_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.o")
            _write_text(p, "JOB ABORTED — CPU LIMIT EXCEEDED\n")
            diag = outdta_failure_diagnosis(p)
            self.assertTrue(diag["failed"])
            self.assertEqual(diag["matched_pattern"], "Job aborted")

    def test_default_pattern_list_has_user_case(self) -> None:
        """守护：用户提供的具体场景必须覆盖。"""
        joined = " | ".join(s.lower() for s in OUTDTA_FAILURE_PATTERNS)
        self.assertIn("transient terminated by failure", joined)
        self.assertIn("errors detected during input processing", joined)

    def test_input_processing_error_signature(self) -> None:
        """RELAP 在输入解析阶段就报错时也必须被识别为失败。"""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.o")
            _write_text(
                p,
                "0******** word  5 on card  20510100 should be in integer format.\n"
                "0******** Data for control component  101 cannot be processed.\n"
                "0******** Errors detected during input processing.\n",
            )
            diag = outdta_failure_diagnosis(p)
            self.assertTrue(diag["failed"])
            self.assertEqual(
                diag["matched_pattern"], "Errors detected during input processing"
            )

    def test_unknown_error_caught_by_prefix_rule(self) -> None:
        """子串未命中、但末行带 ``0********`` 横幅 → 兜底前缀规则判失败。"""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.o")
            _write_text(
                p,
                "0Some interim line\n"
                "0******** unrecognized but obviously bad RELAP banner.\n",
            )
            diag = outdta_failure_diagnosis(p)
            self.assertTrue(diag["failed"])
            self.assertEqual(diag["matched_pattern"], "prefix:0********")

    def test_normal_termination_starting_with_zero_is_safe(self) -> None:
        """前缀规则不能误伤以 ``0`` 开头的正常结束行。"""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.o")
            _write_text(p, "0Transient terminated by end of time step cards.\n")
            diag = outdta_failure_diagnosis(p)
            self.assertFalse(diag["failed"])
            self.assertIsNone(diag["matched_pattern"])

    def test_default_prefix_list_has_known_banners(self) -> None:
        self.assertIn("0********", OUTDTA_FAILURE_PREFIXES)

    def test_default_success_list_has_known_terminal_markers(self) -> None:
        joined = " | ".join(s.lower() for s in OUTDTA_SUCCESS_PATTERNS)
        self.assertIn("end of time step cards", joined)
        self.assertIn("trip", joined)


if __name__ == "__main__":
    unittest.main()
