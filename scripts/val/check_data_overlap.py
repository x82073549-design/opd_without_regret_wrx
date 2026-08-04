#!/usr/bin/env python3
"""Create a reproducible exact-overlap report for training and validation prompts."""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_prompt(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content", value))
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") == "user":
                return str(item.get("content", ""))
        if value:
            return extract_prompt(value[-1])
    return str(value)


def normalized_prompt(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def prompt_records(path: Path, dataset_name: str) -> list[dict]:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        if "prompt" not in frame.columns:
            raise ValueError(f"Dataset has no prompt column: {path}")
        prompt_values = list(frame["prompt"])
    elif path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"JSON dataset must contain a list: {path}")
        prompt_values = [row["prompt"] for row in rows]
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        prompt_values = [row["prompt"] for row in rows]
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

    records = []
    for row_id, value in enumerate(prompt_values):
        prompt = extract_prompt(value).strip()
        records.append(
            {
                "dataset": dataset_name,
                "row_id": row_id,
                "raw_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "normalized_hash": hashlib.sha256(normalized_prompt(prompt).encode("utf-8")).hexdigest(),
            }
        )
    return records


def load_validation_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("purpose") != "validation":
        raise ValueError("Validation manifest purpose must be 'validation'.")
    entries = manifest.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Validation manifest must contain a non-empty datasets list.")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Training parquet file.")
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_path = Path(args.train).resolve()
    manifest_path = Path(args.validation_manifest).resolve()
    output_path = Path(args.output).resolve()

    train_records = prompt_records(train_path, "training")
    train_raw = {record["raw_hash"]: record["row_id"] for record in train_records}
    train_normalized = {record["normalized_hash"]: record["row_id"] for record in train_records}

    validation_datasets = []
    raw_overlaps = []
    normalized_overlaps = []
    for entry in load_validation_manifest(manifest_path):
        validation_path = Path(entry["path"])
        if not validation_path.is_absolute():
            validation_path = manifest_path.parent / validation_path
        validation_path = validation_path.resolve()
        actual_sha256 = sha256_file(validation_path)
        expected_sha256 = entry.get("sha256")
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(
                f"Validation dataset hash mismatch for {validation_path}: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
        records = prompt_records(validation_path, str(entry["name"]))
        validation_datasets.append(
            {
                "name": entry["name"],
                "path": str(validation_path),
                "sha256": actual_sha256,
                "num_questions": len(records),
            }
        )
        for record in records:
            if record["raw_hash"] in train_raw:
                raw_overlaps.append(
                    {
                        "validation_dataset": record["dataset"],
                        "validation_row_id": record["row_id"],
                        "training_row_id": train_raw[record["raw_hash"]],
                        "prompt_hash": record["raw_hash"],
                    }
                )
            if record["normalized_hash"] in train_normalized:
                normalized_overlaps.append(
                    {
                        "validation_dataset": record["dataset"],
                        "validation_row_id": record["row_id"],
                        "training_row_id": train_normalized[record["normalized_hash"]],
                        "prompt_hash": record["normalized_hash"],
                    }
                )

    report = {
        "training_dataset": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "num_questions": len(train_records),
        },
        "validation_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "validation_datasets": validation_datasets,
        "exact_overlap_count": len(raw_overlaps),
        "normalized_text_overlap_count": len(normalized_overlaps),
        "exact_overlaps": raw_overlaps,
        "normalized_text_overlaps": normalized_overlaps,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote overlap report to {output_path}")
    if raw_overlaps or normalized_overlaps:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
