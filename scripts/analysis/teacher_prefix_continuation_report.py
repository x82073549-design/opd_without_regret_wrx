#!/usr/bin/env python3
"""Teacher continuation diagnostic for student-generated prefixes.

This script builds prefix-conditioned prompts from existing student rollout
JSONL files, asks the teacher model to continue, and grades the completed
responses with the same rule-based grader used by the evaluation pipeline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import hashlib
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_UTILS_DIR = ROOT_DIR / "scripts" / "val" / "eval"
if str(EVAL_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_UTILS_DIR))

from utils import grade_answer_verl  # noqa: E402


PREFIX_TYPES = [
    "original_prompt",
    "correct_student_prefix",
    "wrong_student_prefix",
    "random_truncation_prefix",
]

FINAL_ANSWER_PATTERNS = [
    re.compile(r"\\boxed\s*\{", re.IGNORECASE),
    re.compile(r"final\s+answer", re.IGNORECASE),
    re.compile(r"therefore,?\s+the\s+answer", re.IGNORECASE),
    re.compile(r"thus,?\s+the\s+answer", re.IGNORECASE),
    re.compile(r"the\s+answer\s+is", re.IGNORECASE),
]


@dataclass
class SourceRollout:
    task: str
    example_id: int
    prompt: str
    answer: str
    seed: int
    response: str
    correct: bool


def stable_int(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def parse_tasks(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_prefix_fracs(value: str) -> list[float]:
    fracs = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not fracs:
        raise ValueError("--prefix-fracs must contain at least one value.")
    for frac in fracs:
        if frac <= 0.0 or frac >= 1.0:
            raise ValueError(f"prefix fraction must be in (0, 1), got {frac}")
    return fracs


def find_task_file(eval_dir: Path, task: str, max_tokens_tag: str) -> Path:
    matches = sorted(eval_dir.glob(f"{task.lower()}_*{max_tokens_tag}.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No JSONL found for task={task}, tag={max_tokens_tag} under {eval_dir}")
    if len(matches) > 1:
        exact = [path for path in matches if f"-{max_tokens_tag}.jsonl" in path.name]
        if exact:
            matches = exact
    return matches[0]


def load_source_rollouts(eval_dir: Path, tasks: list[str], max_tokens_tag: str) -> list[SourceRollout]:
    rollouts: list[SourceRollout] = []
    for task in tasks:
        path = find_task_file(eval_dir, task, max_tokens_tag)
        print(f"Loading {task} source rollouts from {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                response = str(obj.get("response", ""))
                answer = str(obj["answer"])
                rollouts.append(
                    SourceRollout(
                        task=task.lower(),
                        example_id=int(obj["example_id"]),
                        prompt=str(obj["prompt"]),
                        answer=answer,
                        seed=int(obj.get("seed", -1)),
                        response=response,
                        correct=bool(grade_answer_verl(response, answer)),
                    )
                )
    return rollouts


def first_final_answer_pos(text: str) -> int | None:
    positions = [match.start() for pattern in FINAL_ANSWER_PATTERNS if (match := pattern.search(text))]
    return min(positions) if positions else None


def token_prefix(
    response: str,
    tokenizer: Any,
    frac: float,
    min_prefix_tokens: int,
) -> tuple[str | None, int, str | None]:
    ids = tokenizer.encode(response, add_special_tokens=False)
    if not ids:
        return None, 0, "empty_response"

    raw_len = max(1, int(math.floor(len(ids) * frac)))
    prefix_ids = ids[:raw_len]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)

    leak_pos = first_final_answer_pos(prefix_text)
    if leak_pos is not None:
        prefix_text = prefix_text[:leak_pos].rstrip()
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    if len(prefix_ids) < min_prefix_tokens:
        return None, len(prefix_ids), "prefix_too_short_after_leak_filter"

    return prefix_text, len(prefix_ids), None


def choose_rollout(rows: list[SourceRollout], *seed_parts: Any) -> SourceRollout:
    rng = random.Random(stable_int(*seed_parts))
    return rows[rng.randrange(len(rows))]


def build_requests(
    rollouts: list[SourceRollout],
    tokenizer: Any,
    prefix_fracs: list[float],
    seed: int,
    paired_only: bool,
    max_examples_per_task: int | None,
    min_prefix_tokens: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[SourceRollout]] = defaultdict(list)
    for row in rollouts:
        grouped[(row.task, row.example_id)].append(row)

    by_task: dict[str, list[tuple[tuple[str, int], list[SourceRollout]]]] = defaultdict(list)
    for key, rows in grouped.items():
        by_task[key[0]].append((key, rows))

    requests: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for task, items in sorted(by_task.items()):
        eligible = []
        for key, rows in sorted(items, key=lambda item: item[0][1]):
            correct_rows = [row for row in rows if row.correct]
            wrong_rows = [row for row in rows if not row.correct]
            if paired_only and (not correct_rows or not wrong_rows):
                skipped.append({"task": task, "example_id": key[1], "reason": "not_paired"})
                continue
            if not correct_rows or not wrong_rows:
                skipped.append({"task": task, "example_id": key[1], "reason": "missing_correct_or_wrong"})
                continue
            eligible.append((key, rows, correct_rows, wrong_rows))

        rng = random.Random(stable_int("task_sample", seed, task))
        rng.shuffle(eligible)
        if max_examples_per_task is not None:
            eligible = eligible[:max_examples_per_task]
        eligible.sort(key=lambda item: item[0][1])

        for (task_name, example_id), rows, correct_rows, wrong_rows in eligible:
            base_row = rows[0]
            for frac in prefix_fracs:
                chosen_correct = choose_rollout(correct_rows, seed, task_name, example_id, frac, "correct")
                chosen_wrong = choose_rollout(wrong_rows, seed, task_name, example_id, frac, "wrong")
                wrong_prefix_text, wrong_prefix_len, wrong_skip = token_prefix(
                    chosen_wrong.response, tokenizer, frac, min_prefix_tokens
                )

                def add_request(
                    prefix_type: str,
                    source_row: SourceRollout | None,
                    prefix_text: str,
                    prefix_len: int,
                ) -> None:
                    max_new_tokens = max_tokens - prefix_len
                    if max_new_tokens <= 0:
                        skipped.append(
                            {
                                "task": task_name,
                                "example_id": example_id,
                                "prefix_frac": frac,
                                "prefix_type": prefix_type,
                                "reason": "max_new_tokens_non_positive",
                            }
                        )
                        return
                    request_id = f"{task_name}-{example_id}-f{frac:g}-{prefix_type}"
                    requests.append(
                        {
                            "request_id": request_id,
                            "task": task_name,
                            "example_id": example_id,
                            "answer": base_row.answer,
                            "prefix_type": prefix_type,
                            "prefix_frac": frac,
                            "source_seed": None if source_row is None else source_row.seed,
                            "source_correct": None if source_row is None else source_row.correct,
                            "prefix_token_len": prefix_len,
                            "max_new_tokens": max_new_tokens,
                            "prompt": base_row.prompt,
                            "prefix_text": prefix_text,
                        }
                    )

                add_request("original_prompt", None, "", 0)

                correct_prefix_text, correct_prefix_len, correct_skip = token_prefix(
                    chosen_correct.response, tokenizer, frac, min_prefix_tokens
                )
                if correct_skip is None and correct_prefix_text is not None:
                    add_request("correct_student_prefix", chosen_correct, correct_prefix_text, correct_prefix_len)
                else:
                    skipped.append(
                        {
                            "task": task_name,
                            "example_id": example_id,
                            "prefix_frac": frac,
                            "prefix_type": "correct_student_prefix",
                            "source_seed": chosen_correct.seed,
                            "prefix_token_len": correct_prefix_len,
                            "reason": correct_skip,
                        }
                    )

                if wrong_skip is None and wrong_prefix_text is not None:
                    add_request("wrong_student_prefix", chosen_wrong, wrong_prefix_text, wrong_prefix_len)
                else:
                    skipped.append(
                        {
                            "task": task_name,
                            "example_id": example_id,
                            "prefix_frac": frac,
                            "prefix_type": "wrong_student_prefix",
                            "source_seed": chosen_wrong.seed,
                            "prefix_token_len": wrong_prefix_len,
                            "reason": wrong_skip,
                        }
                    )

                random_row = choose_rollout(rows, seed, task_name, example_id, frac, "random")
                random_ids = tokenizer.encode(random_row.response, add_special_tokens=False)
                target_len = wrong_prefix_len if wrong_prefix_len > 0 else max(1, int(math.floor(len(random_ids) * frac)))
                random_prefix_text = tokenizer.decode(random_ids[:target_len], skip_special_tokens=True)
                leak_pos = first_final_answer_pos(random_prefix_text)
                if leak_pos is not None:
                    random_prefix_text = random_prefix_text[:leak_pos].rstrip()
                random_prefix_len = len(tokenizer.encode(random_prefix_text, add_special_tokens=False))
                if random_prefix_len >= min_prefix_tokens:
                    add_request("random_truncation_prefix", random_row, random_prefix_text, random_prefix_len)
                else:
                    skipped.append(
                        {
                            "task": task_name,
                            "example_id": example_id,
                            "prefix_frac": frac,
                            "prefix_type": "random_truncation_prefix",
                            "source_seed": random_row.seed,
                            "prefix_token_len": random_prefix_len,
                            "reason": "prefix_too_short_after_leak_filter",
                        }
                    )

    return requests, skipped


def split_rollout_ids(rollout_ids: list[int], num_workers: int) -> list[list[int]]:
    chunks = [[] for _ in range(num_workers)]
    for idx, rollout_id in enumerate(rollout_ids):
        chunks[idx % num_workers].append(rollout_id)
    return chunks


def worker_generate(args_tuple: tuple[Any, ...]) -> list[dict[str, Any]]:
    (
        model_path,
        requests,
        rollout_ids,
        gpu_id,
        temperature,
        top_p,
        enable_thinking,
        gpu_memory_utilization,
        tensor_parallel_size,
        generation_batch_size,
    ) = args_tuple

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from vllm import LLM, SamplingParams
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment
    except ImportError:  # pragma: no cover - depends on vLLM version.
        destroy_distributed_environment = None
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
    except ImportError:  # pragma: no cover - depends on vLLM version.
        destroy_model_parallel = None

    llm = None
    outputs: list[dict[str, Any]] = []
    stop_token_ids: list[int] = []
    try:
        print(
            f"[GPU {gpu_id}] loading teacher={model_path}; rollout_ids={rollout_ids}; requests={len(requests)}",
            flush=True,
        )
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )
        tokenizer = llm.get_tokenizer()

        for stop_token in ["<|im_end|>", "<|endoftext|>"]:
            try:
                encoded = tokenizer.encode(stop_token, add_special_tokens=False)
                if encoded:
                    stop_token_ids.append(encoded[0])
            except Exception:
                pass

        formatted_cache: dict[str, str] = {}
        for request in requests:
            formatted_cache[request["request_id"]] = tokenizer.apply_chat_template(
                [{"role": "user", "content": request["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

        batch_size = generation_batch_size if generation_batch_size and generation_batch_size > 0 else len(requests)
        for rollout_id in rollout_ids:
            for start in range(0, len(requests), batch_size):
                batch_requests = requests[start : start + batch_size]
                prompts = [formatted_cache[req["request_id"]] + req["prefix_text"] for req in batch_requests]
                sampling_params = [
                    SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=int(req["max_new_tokens"]),
                        stop_token_ids=stop_token_ids if stop_token_ids else None,
                    )
                    for req in batch_requests
                ]
                generated = llm.generate(prompts, sampling_params, use_tqdm=False)
                for request, out in zip(batch_requests, generated):
                    continuation = out.outputs[0].text
                    outputs.append(
                        {
                            "request_id": request["request_id"],
                            "task": request["task"],
                            "example_id": request["example_id"],
                            "prefix_type": request["prefix_type"],
                            "prefix_frac": request["prefix_frac"],
                            "rollout_id": rollout_id,
                            "continuation": continuation,
                            "full_response": request["prefix_text"] + continuation,
                        }
                    )
    finally:
        print(f"[GPU {gpu_id}] cleaning up", flush=True)
        if llm is not None:
            del llm
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass
        if destroy_distributed_environment is not None:
            try:
                destroy_distributed_environment()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def grade_continuations(
    continuations: list[dict[str, Any]],
    request_map: dict[str, dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    scores = []
    for item in tqdm(continuations, desc="Grading continuations"):
        request = request_map[item["request_id"]]
        full_response = item["full_response"]
        continuation = item["continuation"]
        scores.append(
            {
                "request_id": item["request_id"],
                "task": item["task"],
                "example_id": item["example_id"],
                "prefix_type": item["prefix_type"],
                "prefix_frac": item["prefix_frac"],
                "rollout_id": item["rollout_id"],
                "correct": bool(grade_answer_verl(full_response, request["answer"])),
                "full_response_len": len(tokenizer.encode(full_response, add_special_tokens=False)),
                "continuation_len": len(tokenizer.encode(continuation, add_special_tokens=False)),
                "prefix_token_len": request["prefix_token_len"],
                "format_error": "boxed" not in full_response,
            }
        )
    return scores


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def per_request_metrics(scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        grouped[row["request_id"]].append(row)

    metrics = {}
    for request_id, rows in grouped.items():
        corrects = [1.0 if row["correct"] else 0.0 for row in rows]
        first = rows[0]
        metrics[request_id] = {
            "request_id": request_id,
            "task": first["task"],
            "example_id": first["example_id"],
            "prefix_type": first["prefix_type"],
            "prefix_frac": float(first["prefix_frac"]),
            "avg": mean(corrects),
            "pass": 1.0 if any(corrects) else 0.0,
            "solve_all": 1.0 if all(corrects) else 0.0,
            "solve_none": 1.0 if not any(corrects) else 0.0,
            "num_rollouts": len(rows),
            "avg_continuation_tokens": mean([float(row["continuation_len"]) for row in rows]),
            "format_errors": sum(1 for row in rows if row["format_error"]),
        }
    return metrics


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    boots = []
    n = len(values)
    for _ in range(samples):
        boots.append(mean([values[rng.randrange(n)] for _ in range(n)]))
    boots.sort()
    low_idx = int(math.floor(0.025 * (len(boots) - 1)))
    high_idx = int(math.floor(0.975 * (len(boots) - 1)))
    return boots[low_idx], boots[high_idx]


def summarize(
    requests: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    request_map = {row["request_id"]: row for row in requests}
    request_metrics = per_request_metrics(scores)

    group_keys: list[tuple[str, str, float, str]] = []
    for group_name in ["overall"] + sorted({row["task"] for row in requests}):
        task_filter = None if group_name == "overall" else group_name
        for frac in sorted({float(row["prefix_frac"]) for row in requests}):
            for prefix_type in PREFIX_TYPES:
                group_keys.append((group_name, task_filter or "", frac, prefix_type))

    summary_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    for group_name, task_filter, frac, prefix_type in group_keys:
        req_ids = [
            row["request_id"]
            for row in requests
            if float(row["prefix_frac"]) == frac
            and row["prefix_type"] == prefix_type
            and (not task_filter or row["task"] == task_filter)
            and row["request_id"] in request_metrics
        ]
        if not req_ids:
            continue
        req_metrics = [request_metrics[req_id] for req_id in req_ids]
        req_objs = [request_map[req_id] for req_id in req_ids]
        num_rollouts = sum(int(metric["num_rollouts"]) for metric in req_metrics)
        row = {
            "group": group_name,
            "prefix_frac": frac,
            "prefix_type": prefix_type,
            "num_examples": len(req_ids),
            "num_rollouts": num_rollouts,
            "avg@k": mean([metric["avg"] for metric in req_metrics]),
            "pass@k": mean([metric["pass"] for metric in req_metrics]),
            "solve_none": sum(int(metric["solve_none"]) for metric in req_metrics),
            "solve_all": sum(int(metric["solve_all"]) for metric in req_metrics),
            "avg_prefix_tokens": mean([float(req["prefix_token_len"]) for req in req_objs]),
            "avg_continuation_tokens": mean([float(metric["avg_continuation_tokens"]) for metric in req_metrics]),
            "format_error_rollouts": sum(int(metric["format_errors"]) for metric in req_metrics),
            "pass_delta_wrong_vs_original": "",
            "pass_delta_wrong_vs_correct": "",
            "pass_delta_wrong_vs_random": "",
            "avg_delta_wrong_vs_original": "",
            "avg_delta_wrong_vs_correct": "",
            "avg_delta_wrong_vs_random": "",
            "bootstrap_ci_low": "",
            "bootstrap_ci_high": "",
        }
        summary_rows.append(row)

    keyed_metrics: dict[tuple[str, int, float, str], dict[str, Any]] = {}
    for metric in request_metrics.values():
        keyed_metrics[(metric["task"], int(metric["example_id"]), float(metric["prefix_frac"]), metric["prefix_type"])] = metric

    for group_name in ["overall"] + sorted({row["task"] for row in requests}):
        for frac in sorted({float(row["prefix_frac"]) for row in requests}):
            tasks = sorted({row["task"] for row in requests}) if group_name == "overall" else [group_name]
            example_keys = sorted(
                {
                    (task, int(row["example_id"]))
                    for row in requests
                    for task in tasks
                    if row["task"] == task and float(row["prefix_frac"]) == frac
                }
            )
            paired = []
            for task, example_id in example_keys:
                key_base = (task, example_id, frac)
                needed = [keyed_metrics.get((*key_base, prefix_type)) for prefix_type in PREFIX_TYPES]
                if all(item is not None for item in needed):
                    paired.append((task, example_id, needed))
            if not paired:
                continue

            deltas = {
                "pass_delta_wrong_vs_original": [],
                "pass_delta_wrong_vs_correct": [],
                "pass_delta_wrong_vs_random": [],
                "avg_delta_wrong_vs_original": [],
                "avg_delta_wrong_vs_correct": [],
                "avg_delta_wrong_vs_random": [],
            }
            for _, _, items in paired:
                by_type = {item["prefix_type"]: item for item in items if item is not None}
                wrong = by_type["wrong_student_prefix"]
                original = by_type["original_prompt"]
                correct = by_type["correct_student_prefix"]
                random_prefix = by_type["random_truncation_prefix"]
                deltas["pass_delta_wrong_vs_original"].append(wrong["pass"] - original["pass"])
                deltas["pass_delta_wrong_vs_correct"].append(wrong["pass"] - correct["pass"])
                deltas["pass_delta_wrong_vs_random"].append(wrong["pass"] - random_prefix["pass"])
                deltas["avg_delta_wrong_vs_original"].append(wrong["avg"] - original["avg"])
                deltas["avg_delta_wrong_vs_correct"].append(wrong["avg"] - correct["avg"])
                deltas["avg_delta_wrong_vs_random"].append(wrong["avg"] - random_prefix["avg"])

            ci_low, ci_high = bootstrap_ci(
                deltas["pass_delta_wrong_vs_correct"], bootstrap_samples, stable_int(seed, group_name, frac, "ci")
            )
            delta_row = {
                "group": group_name,
                "prefix_frac": frac,
                "paired_examples": len(paired),
                **{name: mean(values) for name, values in deltas.items()},
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
            delta_rows.append(delta_row)

            for row in summary_rows:
                if row["group"] == group_name and float(row["prefix_frac"]) == frac and row["prefix_type"] == "wrong_student_prefix":
                    for name, values in deltas.items():
                        row[name] = mean(values)
                    row["bootstrap_ci_low"] = ci_low
                    row["bootstrap_ci_high"] = ci_high

    return summary_rows, delta_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def write_summary_md(path: Path, summary_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> None:
    lines = ["# Teacher Prefix Continuation Report", ""]

    lines.append("## Prefix Results")
    lines.append("")
    lines.append("| group | frac | prefix_type | examples | avg@k | pass@k | avg_prefix_tokens | avg_cont_tokens |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {group} | {prefix_frac} | {prefix_type} | {num_examples} | {avg} | {passk} | {prefix} | {cont} |".format(
                group=row["group"],
                prefix_frac=fmt(row["prefix_frac"]),
                prefix_type=row["prefix_type"],
                num_examples=row["num_examples"],
                avg=fmt(row["avg@k"]),
                passk=fmt(row["pass@k"]),
                prefix=fmt(row["avg_prefix_tokens"]),
                cont=fmt(row["avg_continuation_tokens"]),
            )
        )

    lines.append("")
    lines.append("## Paired Deltas")
    lines.append("")
    lines.append(
        "| group | frac | pairs | pass wrong-original | pass wrong-correct | pass wrong-random | avg wrong-correct | CI pass wrong-correct |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in delta_rows:
        lines.append(
            "| {group} | {frac} | {pairs} | {d0} | {d1} | {d2} | {ad1} | [{lo}, {hi}] |".format(
                group=row["group"],
                frac=fmt(row["prefix_frac"]),
                pairs=row["paired_examples"],
                d0=fmt(row["pass_delta_wrong_vs_original"]),
                d1=fmt(row["pass_delta_wrong_vs_correct"]),
                d2=fmt(row["pass_delta_wrong_vs_random"]),
                ad1=fmt(row["avg_delta_wrong_vs_correct"]),
                lo=fmt(row["bootstrap_ci_low"]),
                hi=fmt(row["bootstrap_ci_high"]),
            )
        )

    lines.append("")
    lines.append("## Diagnostic Notes")
    lines.append("")
    overall = [row for row in delta_rows if row["group"] == "overall"]
    if not overall:
        lines.append("- No fully paired examples were available for the requested comparison.")
    else:
        for row in overall:
            frac = row["prefix_frac"]
            d_correct = row["pass_delta_wrong_vs_correct"]
            d_original = row["pass_delta_wrong_vs_original"]
            d_random = row["pass_delta_wrong_vs_random"]
            if d_correct <= -0.10 and d_original <= -0.05:
                verdict = "supports wrong-prefix teacher unreliability"
            elif d_correct <= -0.10 and d_random > -0.05:
                verdict = "mostly shows correct prefixes help; wrong prefixes are not clearly worse than random control"
            elif abs(d_correct) < 0.05:
                verdict = "does not support a clear correct-vs-wrong prefix gap"
            elif d_random <= -0.10:
                verdict = "random control also drops; length/format effects may confound the claim"
            else:
                verdict = "mixed evidence"
            lines.append(
                f"- frac={fmt(frac)}: {verdict} "
                f"(pass wrong-correct={fmt(d_correct)}, wrong-original={fmt(d_original)}, wrong-random={fmt(d_random)})."
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose teacher continuation under student prefixes.")
    parser.add_argument("--source-eval-dir", default="justrl_eval_outputs/global_step_279_hf")
    parser.add_argument("--teacher-model-path", default="model/JustRL-DeepSeek-1.5B")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tasks", default="amc23,aime24,aime25")
    parser.add_argument("--max-tokens-tag", default="MNT31744")
    parser.add_argument("--prefix-fracs", default="0.25,0.5,0.75")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=31744)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--thinking-flag", choices=["disable", "enable"], default="disable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paired-only", dest="paired_only", action="store_true", default=True)
    parser.add_argument("--no-paired-only", dest="paired_only", action="store_false")
    parser.add_argument("--max-examples-per-task", type=int, default=None)
    parser.add_argument("--min-prefix-tokens", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--skip-generation", action="store_true", help="Reuse continuations.jsonl if present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_eval_dir = Path(args.source_eval_dir)
    teacher_model_path = Path(args.teacher_model_path)
    out_dir = Path(args.out_dir)
    tasks = parse_tasks(args.tasks)
    prefix_fracs = parse_prefix_fracs(args.prefix_fracs)
    gpu_workers = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_workers:
        raise ValueError("--gpus must contain at least one GPU id.")

    if out_dir.exists() and args.replace:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_files = [
        out_dir / "continuation_requests.jsonl",
        out_dir / "continuations.jsonl",
        out_dir / "rollout_scores.jsonl",
        out_dir / "summary.csv",
        out_dir / "summary.md",
        out_dir / "config.json",
    ]
    if not args.replace and not args.skip_generation:
        existing = [path for path in output_files if path.exists()]
        if existing:
            raise FileExistsError(f"Output files already exist; pass --replace to overwrite: {existing}")

    config = vars(args).copy()
    config["tasks"] = tasks
    config["prefix_fracs"] = prefix_fracs
    config["source_eval_dir"] = str(source_eval_dir)
    config["teacher_model_path"] = str(teacher_model_path)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Loading teacher tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(teacher_model_path), trust_remote_code=True, local_files_only=True)

    if args.skip_generation:
        requests = read_jsonl(out_dir / "continuation_requests.jsonl")
        continuations = read_jsonl(out_dir / "continuations.jsonl")
    else:
        rollouts = load_source_rollouts(source_eval_dir, tasks, args.max_tokens_tag)
        print(f"Loaded {len(rollouts)} source rollouts")

        requests, skipped = build_requests(
            rollouts=rollouts,
            tokenizer=tokenizer,
            prefix_fracs=prefix_fracs,
            seed=args.seed,
            paired_only=args.paired_only,
            max_examples_per_task=args.max_examples_per_task,
            min_prefix_tokens=args.min_prefix_tokens,
            max_tokens=args.max_tokens,
        )
        print(f"Built {len(requests)} continuation requests; skipped {len(skipped)} candidates")
        write_jsonl(out_dir / "continuation_requests.jsonl", requests)
        write_jsonl(out_dir / "skipped_requests.jsonl", skipped)

        rollout_ids = list(range(args.n))
        rollout_chunks = split_rollout_ids(rollout_ids, len(gpu_workers))
        worker_args = [
            (
                str(teacher_model_path),
                requests,
                rollout_chunks[idx],
                gpu_workers[idx],
                args.temperature,
                args.top_p,
                args.thinking_flag == "enable",
                args.gpu_memory_utilization,
                args.tensor_parallel_size,
                args.generation_batch_size,
            )
            for idx in range(len(gpu_workers))
            if rollout_chunks[idx]
        ]

        continuations = []
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=ctx) as ex:
            futures = [ex.submit(worker_generate, item) for item in worker_args]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="GPU workers"):
                continuations.extend(fut.result())

        continuations.sort(key=lambda row: (row["task"], row["example_id"], row["prefix_frac"], row["prefix_type"], row["rollout_id"]))
        write_jsonl(out_dir / "continuations.jsonl", continuations)

    request_map = {row["request_id"]: row for row in requests}
    scores = grade_continuations(continuations, request_map, tokenizer)
    scores.sort(key=lambda row: (row["task"], row["example_id"], row["prefix_frac"], row["prefix_type"], row["rollout_id"]))
    write_jsonl(out_dir / "rollout_scores.jsonl", scores)

    summary_rows, delta_rows = summarize(scores=scores, requests=requests, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "paired_deltas.csv", delta_rows)
    write_summary_md(out_dir / "summary.md", summary_rows, delta_rows)

    print(f"Report written to {out_dir}")
    print(f"Summary: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
