#!/usr/bin/env python3
"""Compare candidate and baseline detailed evaluation outputs with paired bootstrap."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


PAIR_FIELDS = ("task", "question_id", "evaluation_seed")


def load_rows(path: Path) -> dict[tuple, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = tuple(row[field] for field in PAIR_FIELDS)
            if key in rows:
                raise ValueError(f"Duplicate paired-evaluation key in {path}: {key}")
            rows[key] = row
    if not rows:
        raise ValueError(f"No detailed evaluation rows found in {path}")
    return rows


def percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    candidate_path = Path(args.candidate).resolve()
    baseline_path = Path(args.baseline).resolve()
    candidate = load_rows(candidate_path)
    baseline = load_rows(baseline_path)
    if candidate.keys() != baseline.keys():
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_baseline = sorted(set(candidate) - set(baseline))
        raise ValueError(
            "Candidate and baseline do not have identical paired keys. "
            f"Missing candidate={missing_candidate[:5]}, missing baseline={missing_baseline[:5]}"
        )

    deltas = {
        key: float(candidate[key]["correct"]) - float(baseline[key]["correct"])
        for key in candidate
    }
    question_groups = defaultdict(list)
    for (task, question_id, evaluation_seed), delta in deltas.items():
        question_groups[(task, question_id)].append((evaluation_seed, delta))

    groups = sorted(question_groups)
    rng = random.Random(args.bootstrap_seed)
    bootstrap_means = []
    for _ in range(args.bootstrap_samples):
        sampled_groups = [groups[rng.randrange(len(groups))] for _ in groups]
        sampled_deltas = [delta for group in sampled_groups for _, delta in question_groups[group]]
        bootstrap_means.append(mean(sampled_deltas))
    bootstrap_means.sort()

    evaluation_seeds = sorted({key[2] for key in deltas})
    midpoint = max(1, len(evaluation_seeds) // 2)
    seed_lists = [evaluation_seeds[:midpoint], evaluation_seeds[midpoint:]]
    seed_list_results = []
    for seed_list in seed_lists:
        if not seed_list:
            continue
        seed_set = set(seed_list)
        seed_deltas = [delta for key, delta in deltas.items() if key[2] in seed_set]
        seed_list_results.append(
            {
                "evaluation_seeds": seed_list,
                "mean_paired_delta": mean(seed_deltas),
                "num_pairs": len(seed_deltas),
            }
        )

    report = {
        "candidate_file": str(candidate_path),
        "baseline_file": str(baseline_path),
        "pair_fields": list(PAIR_FIELDS),
        "num_questions": len(groups),
        "num_pairs": len(deltas),
        "candidate_mean": mean([float(row["correct"]) for row in candidate.values()]),
        "baseline_mean": mean([float(row["correct"]) for row in baseline.values()]),
        "mean_paired_delta": mean(list(deltas.values())),
        "paired_bootstrap": {
            "unit": "question",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "confidence_interval_95": [
                percentile(bootstrap_means, 0.025),
                percentile(bootstrap_means, 0.975),
            ],
        },
        "evaluation_seed_lists": seed_list_results,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote paired evaluation report to {output_path}")


if __name__ == "__main__":
    main()
