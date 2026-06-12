#!/usr/bin/env python3
"""Collect OPD eval grading results into a compact comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK_ORDER = ("amc23", "aime24", "aime25")


def load_result(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list")
    by_task = {}
    for item in data:
        hp = item.get("hyperparameters", {})
        task = str(hp.get("task_name", "")).lower()
        if task:
            by_task[task] = item
    return by_task


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def make_run_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parent = rel.parent
    return str(parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="justrl_eval_outputs")
    parser.add_argument("--contains", default="", help="Only keep runs whose path contains this substring.")
    parser.add_argument("--markdown", action="store_true", help="Print a markdown table.")
    args = parser.parse_args()

    root = Path(args.eval_root)
    rows = []
    for path in sorted(root.rglob("grading_results.json")):
        run_name = make_run_name(path, root)
        if args.contains and args.contains not in run_name:
            continue
        try:
            by_task = load_result(path)
        except Exception as exc:  # noqa: BLE001 - diagnostic script should keep going.
            print(f"skip {path}: {exc}")
            continue
        row = {"run": run_name}
        for task in TASK_ORDER:
            item = by_task.get(task, {})
            row[f"{task}_avg"] = fmt_float(item.get("mean_score"))
            row[f"{task}_pass"] = fmt_float(item.get("best_score"))
            row[f"{task}_none"] = item.get("solve_none", "")
            row[f"{task}_len"] = fmt_float(item.get("avg_output_length"), 1)
        rows.append(row)

    headers = ["run"]
    for task in TASK_ORDER:
        headers.extend([f"{task}_avg", f"{task}_pass", f"{task}_none", f"{task}_len"])

    if args.markdown:
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            print("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    else:
        print("\t".join(headers))
        for row in rows:
            print("\t".join(str(row.get(h, "")) for h in headers))


if __name__ == "__main__":
    main()
