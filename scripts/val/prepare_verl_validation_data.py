import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROMPT_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def add_prompt_suffix(prompt_value):
    prompt_items = copy.deepcopy(list(prompt_value))
    if not prompt_items:
        return np.array(prompt_items, dtype=object)
    content = str(prompt_items[0].get("content", "")).strip()
    if not content.endswith(PROMPT_SUFFIX):
        content = f"{content}{PROMPT_SUFFIX}"
    prompt_items[0]["content"] = content
    return np.array(prompt_items, dtype=object)


def prepare_task(source_dir: Path, output_dir: Path, task: str) -> Path:
    src_path = source_dir / task / "test.parquet"
    if not src_path.exists():
        raise FileNotFoundError(f"Missing source validation file: {src_path}")

    df = pd.read_parquet(src_path)
    if "prompt" not in df.columns:
        raise ValueError(f"{src_path} does not contain a prompt column")

    df = df.copy()
    df["prompt"] = df["prompt"].map(add_prompt_suffix)

    dst_task_dir = output_dir / task
    dst_task_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_task_dir / "test.parquet"
    df.to_parquet(dst_path, index=False)

    first_row_path = dst_task_dir / "test.parquet_first_row.json"
    with first_row_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(df.iloc[0].to_dict()), f, indent=2, ensure_ascii=False)

    return dst_path


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description="Prepare scripts/val data for verl-side validation.")
    parser.add_argument("--source-dir", default="scripts/val/data")
    parser.add_argument("--output-dir", default="scripts/val/data_verl")
    parser.add_argument("--tasks", nargs="+", default=["AIME25", "AMC23", "AIME24"])
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    for task in args.tasks:
        dst_path = prepare_task(source_dir, output_dir, task)
        print(dst_path)


if __name__ == "__main__":
    main()
