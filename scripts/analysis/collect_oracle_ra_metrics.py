#!/usr/bin/env python3
"""Summarize Oracle RA-OPD training metrics from console logs."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path


FIELDS = (
    "oracle_ra/rho_25_mean",
    "oracle_ra/rho_75_mean",
    "oracle_ra/gate_mean",
    "oracle_ra/gate_low_ratio_0.5",
    "oracle_ra/correct_gate_mean",
    "oracle_ra/wrong_gate_mean",
    "critic/true_reward/mean",
    "response_length/clip_ratio",
    "timing_s/oracle_ra_opd",
    "timing_s/step",
)


def parse_log(path: Path) -> list[dict[str, float]]:
    text = path.read_text(errors="ignore")
    rows = []
    for match in re.finditer(r"step:(\d+) - (.*?)(?=\n)", text):
        line = match.group(2)
        row: dict[str, float] = {"step": float(match.group(1))}
        for field in FIELDS:
            value_match = re.search(re.escape(field) + r":([-+0-9.eE]+)", line)
            if value_match:
                row[field] = float(value_match.group(1))
        if "oracle_ra/gate_mean" in row:
            correct = row.get("oracle_ra/correct_gate_mean")
            wrong = row.get("oracle_ra/wrong_gate_mean")
            if correct is not None and wrong is not None:
                row["oracle_ra/correct_wrong_gate_gap"] = correct - wrong
            rows.append(row)
    return rows


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def summarize(rows: list[dict[str, float]], field: str) -> tuple[str, str, str, str]:
    values = [row[field] for row in rows if field in row]
    if not values:
        return "", "", "", ""
    return fmt(statistics.mean(values)), fmt(values[-1]), fmt(min(values)), fmt(max(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--last", type=int, default=0, help="Only summarize the last N steps.")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    rows = parse_log(args.log)
    if args.last > 0:
        rows = rows[-args.last :]

    if not rows:
        raise SystemExit(f"No Oracle RA metrics found in {args.log}")

    latest = int(rows[-1]["step"])
    fields = (
        "oracle_ra/rho_25_mean",
        "oracle_ra/rho_75_mean",
        "oracle_ra/gate_mean",
        "oracle_ra/gate_low_ratio_0.5",
        "oracle_ra/correct_gate_mean",
        "oracle_ra/wrong_gate_mean",
        "oracle_ra/correct_wrong_gate_gap",
        "critic/true_reward/mean",
        "response_length/clip_ratio",
        "timing_s/oracle_ra_opd",
        "timing_s/step",
    )

    if args.markdown:
        print(f"latest_step: {latest}")
        print()
        print("| metric | mean | last | min | max |")
        print("| --- | --- | --- | --- | --- |")
        for field in fields:
            print("| " + " | ".join((field, *summarize(rows, field))) + " |")
    else:
        print(f"latest_step\t{latest}")
        print("metric\tmean\tlast\tmin\tmax")
        for field in fields:
            print("\t".join((field, *summarize(rows, field))))


if __name__ == "__main__":
    main()
