"""
RELAP5 ``outdta.o`` 末行失败检测。

RELAP 在每次瞬态计算结束时都会在 ``outdta.o`` 末行写一条
``Transient terminated by ...`` 总结：

- ``Transient terminated by end of time step cards.``  → 正常结束
- ``Transient terminated by trip.``                    → 正常结束（触发器）
- ``Transient terminated by failure.``                 → **失败**（参数越界 / 数值发散等）

更宽泛的失败签名见 :data:`OUTDTA_FAILURE_PATTERNS`。本模块仅提供检测结果，不
触发任何重试，由调用方（``auto_optimize``）决定后续动作。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

# 大小写不敏感的子串匹配；命中任一即视为失败。
OUTDTA_FAILURE_PATTERNS: List[str] = [
    # 瞬态计算阶段的失败终止
    "Transient terminated by failure",
    "Transient terminated abnormally",
    # 输入解析 / 初值阶段的失败（连 transient 都没跑起来）
    "Errors detected during input processing",
    "Errors detected during initial conditions",
    "Errors detected during major edit",
    "Errors detected during steady state",
    "should be in integer format",
    "should be in real format",
    "cannot be processed",
    # 通用的 abort / fatal
    "Code aborted",
    "Job aborted",
    "Errtrm",
    "fatal error",
]

# RELAP runs that are acceptable optimization samples must finish with one of
# these terminal summaries.  Some Fortran run-time errors only appear on stdout
# while ``outdta.o`` ends with the last diagnostic/data line, so absence of a
# normal terminal marker is treated as a failed run by default.
OUTDTA_SUCCESS_PATTERNS: List[str] = [
    "Transient terminated by end of time step cards",
    "Transient terminated by trip",
]

# 行首结构化前缀：末行以这些前缀开头时判失败（兜底，覆盖未列入子串的新句式）。
# RELAP 用 ``0`` 作 FORTRAN carriage-control（"换页"），后接星号 / 美元号是错误级横幅。
# 正常结束的末行通常是 ``0Transient terminated by ...``（没有 ``********``）。
OUTDTA_FAILURE_PREFIXES: List[str] = [
    "0********",
    "0$$$$$$$",
]


def read_last_nonempty_line(path: str, max_bytes: int = 8192) -> str:
    """
    从尾部最多读取 ``max_bytes`` 字节并返回最后一个非空行（去除前后空白与换行）。

    设计目标：单次系统调用、不加载整个 ``outdta.o``（典型几 MB 至几十 MB），
    亦能正确处理 Windows ``CRLF`` / Unix ``LF``。
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    if size <= 0:
        return ""
    n = min(int(max_bytes), size)
    try:
        with open(path, "rb") as f:
            f.seek(size - n)
            chunk = f.read(n)
    except OSError:
        return ""

    text = chunk.decode("utf-8", errors="ignore")
    for line in reversed(text.splitlines()):
        s = line.rstrip("\r\n").strip()
        if s:
            return s
    return ""


def read_tail_lines(path: str, max_lines: int = 20, max_bytes: int = 32768) -> List[str]:
    """从尾部读取最多 ``max_bytes`` 字节并返回最后 ``max_lines`` 个非空行。"""
    try:
        size = os.path.getsize(path)
    except OSError:
        return []
    if size <= 0:
        return []
    n = min(int(max_bytes), size)
    try:
        with open(path, "rb") as f:
            f.seek(size - n)
            chunk = f.read(n)
    except OSError:
        return []

    text = chunk.decode("utf-8", errors="ignore")
    lines = [s.rstrip("\r\n") for s in text.splitlines()]
    nonempty = [s for s in lines if s.strip()]
    if max_lines <= 0 or len(nonempty) <= max_lines:
        return nonempty
    return nonempty[-int(max_lines):]


def outdta_failure_diagnosis(
    path: str,
    patterns: Optional[List[str]] = None,
    prefixes: Optional[List[str]] = None,
    max_bytes: int = 8192,
) -> Dict[str, object]:
    """
    末行检测：返回是否失败、命中模式、末行原文。

    判定顺序：

    1. 子串匹配（``patterns`` / :data:`OUTDTA_FAILURE_PATTERNS`，大小写不敏感）
    2. 行首前缀匹配（``prefixes`` / :data:`OUTDTA_FAILURE_PREFIXES`）—— 兜底规则

    Returns
    -------
    dict
      - ``exists``         : ``outdta.o`` 是否存在
      - ``last_line``      : 末非空行（找不到时空串）
      - ``failed``         : 命中任一规则时为 True
      - ``matched_pattern``: 命中的子串；前缀命中时返回 ``"prefix:<前缀>"``
    """
    out: Dict[str, object] = {
        "exists": os.path.isfile(path),
        "last_line": "",
        "failed": False,
        "matched_pattern": None,
    }
    if not out["exists"]:
        return out

    last = read_last_nonempty_line(path, max_bytes=max_bytes)
    out["last_line"] = last
    if not last:
        out["failed"] = True
        out["matched_pattern"] = "missing_normal_termination"
        return out

    tail_lines = read_tail_lines(path, max_lines=40, max_bytes=max(32768, max_bytes))
    tail_text = "\n".join(tail_lines)

    pats = patterns if patterns is not None else OUTDTA_FAILURE_PATTERNS
    low = tail_text.lower()
    for p in pats:
        if str(p).lower() in low:
            out["failed"] = True
            out["matched_pattern"] = p
            return out

    pres = prefixes if prefixes is not None else OUTDTA_FAILURE_PREFIXES
    for prefix in pres:
        if last.startswith(prefix):
            out["failed"] = True
            out["matched_pattern"] = f"prefix:{prefix}"
            return out

    success_low = tail_text.lower()
    for p in OUTDTA_SUCCESS_PATTERNS:
        if p.lower() in success_low:
            return out

    out["failed"] = True
    out["matched_pattern"] = "missing_normal_termination"

    return out
