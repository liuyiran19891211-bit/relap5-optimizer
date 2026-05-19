#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract RELAP stripf plotrec blocks into one-row-per-timestep CSV.

Usage:
    python extract_plotrec_rows.py --input stripf --output stripf_plotrec_rows.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def _parse_float_tokens(tokens: List[str]) -> List[float]:
    values: List[float] = []
    for tok in tokens:
        values.append(float(tok))
    return values


def parse_stripf_headers(stripf_path: Path) -> Tuple[List[str], List[str]]:
    """
    Parse column names and column IDs from stripf header section.

    Expected layout in stripf:
    - plotalf ... (may continue to next line(s))
    - plotnum ... (may continue to next line(s))
    - then plotrec starts
    """
    with stripf_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    names: List[str] = []
    nums: List[str] = []

    i = 0
    total = len(lines)
    while i < total:
        s = lines[i].strip()
        low = s.lower()

        if low.startswith("plotalf"):
            names.extend(s.split()[1:])
            i += 1
            while i < total:
                t = lines[i].strip()
                t_low = t.lower()
                if not t or t_low.startswith(("plotnum", "plotrec", "plotinf")):
                    break
                names.extend(t.split())
                i += 1
            continue

        if low.startswith("plotnum"):
            nums.extend(s.split()[1:])
            i += 1
            while i < total:
                t = lines[i].strip()
                t_low = t.lower()
                if not t or t_low.startswith(("plotrec", "plotalf", "plotinf")):
                    break
                nums.extend(t.split())
                i += 1
            break

        i += 1

    return names, nums


def parse_plotrec_rows(stripf_path: Path) -> List[List[float]]:
    """
    Parse all plotrec records from stripf.

    Rule:
    - A new time step starts at a line beginning with "plotrec".
    - Remaining data for that time step are continuation lines until next "plotrec".
    - Continuation line is accepted only when every token can be parsed as float.
    """
    rows: List[List[float]] = []
    current: List[float] | None = None

    with stripf_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.lower().startswith("plotrec"):
                # Close previous record if any.
                if current:
                    rows.append(current)

                rest = line[len("plotrec") :].strip()
                if not rest:
                    current = []
                    continue

                try:
                    current = _parse_float_tokens(rest.split())
                except ValueError:
                    current = []
                continue

            # Continuation line for current plotrec record.
            if current is not None:
                parts = line.split()
                try:
                    current.extend(_parse_float_tokens(parts))
                except ValueError:
                    # Non-numeric line means this is not a continuation line.
                    pass

    if current:
        rows.append(current)

    return rows


def build_csv_header(
    names: List[str], nums: List[str], max_len: int
) -> List[str]:
    """
    Build final CSV header from plotalf names + plotnum IDs.
    """
    header: List[str] = []
    for idx in range(max_len):
        name = names[idx] if idx < len(names) else f"col{idx}"
        num = nums[idx] if idx < len(nums) else ""
        if num:
            header.append(f"{name}_{num}")
        else:
            header.append(name)
    return header


def write_rows_to_csv(
    rows: List[List[float]], output_path: Path, header: List[str]
) -> None:
    if not rows:
        raise RuntimeError("No plotrec data found.")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        max_len = len(header)
        for row in rows:
            padded = [f"{v:.9E}" for v in row] + [""] * (max_len - len(row))
            writer.writerow(padded)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract each plotrec timestep into one CSV row."
    )
    parser.add_argument(
        "--input",
        default="stripf",
        help="Path to RELAP stripf file (default: stripf)",
    )
    parser.add_argument(
        "--output",
        default="stripf_plotrec_rows.csv",
        help="Output CSV file path (default: stripf_plotrec_rows.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    names, nums = parse_stripf_headers(input_path)
    rows = parse_plotrec_rows(input_path)
    max_cols = max(len(r) for r in rows) if rows else 0
    header = build_csv_header(names, nums, max_cols)
    write_rows_to_csv(rows, output_path, header)

    print(f"Extracted {len(rows)} timesteps.")
    print(f"Output file: {output_path.resolve()}")
    print(f"Columns (max): {max_cols}")


if __name__ == "__main__":
    main()
