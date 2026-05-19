"""Unit tests for auto_optimize.run_bat batch wrapper behavior."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import auto_optimize


class TestRunBatWrapper(unittest.TestCase):
    def test_run_bat_uses_explicit_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "start.bat")
            artifact = os.path.join(tmp, "outdta")
            with open(script, "w", encoding="utf-8") as f:
                f.write("@echo off\n")

            def fake_run(*args, **kwargs):
                self.assertEqual(os.path.abspath(kwargs["cwd"]), os.path.abspath(tmp))
                with open(artifact, "w", encoding="utf-8") as f:
                    f.write("fresh RELAP output\n")
                return subprocess.CompletedProcess(
                    args=args[0],
                    returncode=0,
                    stdout="ok\n",
                )

            with mock.patch("auto_optimize.subprocess.run", side_effect=fake_run):
                auto_optimize.run_bat(
                    "start.bat",
                    cwd=tmp,
                    expect_file_after="outdta",
                    min_expected_bytes_after=4,
                    heartbeat_interval_sec=0,
                )

    def test_nonzero_exit_is_warning_when_artifact_is_refreshed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_base = auto_optimize.BASE_DIR
            auto_optimize.BASE_DIR = tmp
            try:
                script = os.path.join(tmp, "start.bat")
                artifact = os.path.join(tmp, "outdta")
                with open(script, "w", encoding="utf-8") as f:
                    f.write("@echo off\n")

                def fake_run(*args, **kwargs):
                    with open(artifact, "w", encoding="utf-8") as f:
                        f.write("fresh RELAP output\n")
                    return subprocess.CompletedProcess(
                        args=args[0],
                        returncode=1,
                        stdout="batch returned 1\n",
                    )

                logs = []
                with mock.patch("auto_optimize.subprocess.run", side_effect=fake_run):
                    elapsed = auto_optimize.run_bat(
                        "start.bat",
                        emit=logs.append,
                        expect_file_after="outdta",
                        min_expected_bytes_after=4,
                        heartbeat_interval_sec=0,
                    )

                self.assertGreaterEqual(elapsed, 0.0)
                self.assertTrue(any("返回退出码 1" in line for line in logs))
                self.assertTrue(any("结束，退出码 1" in line for line in logs))
            finally:
                auto_optimize.BASE_DIR = old_base

    def test_stale_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_base = auto_optimize.BASE_DIR
            auto_optimize.BASE_DIR = tmp
            try:
                script = os.path.join(tmp, "start.bat")
                artifact = os.path.join(tmp, "outdta")
                with open(script, "w", encoding="utf-8") as f:
                    f.write("@echo off\n")
                with open(artifact, "w", encoding="utf-8") as f:
                    f.write("old RELAP output\n")
                old_ts = time.time() - 60.0
                os.utime(artifact, (old_ts, old_ts))

                completed = subprocess.CompletedProcess(
                    args=["cmd.exe"], returncode=1, stdout="failed early\n"
                )
                with mock.patch(
                    "auto_optimize.subprocess.run", return_value=completed
                ):
                    with self.assertRaisesRegex(RuntimeError, "未在本次运行中刷新"):
                        auto_optimize.run_bat(
                            "start.bat",
                            emit=lambda _msg: None,
                            expect_file_after="outdta",
                            min_expected_bytes_after=4,
                            heartbeat_interval_sec=0,
                        )
            finally:
                auto_optimize.BASE_DIR = old_base


if __name__ == "__main__":
    unittest.main()
