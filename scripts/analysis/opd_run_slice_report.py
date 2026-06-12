#!/usr/bin/env python3
"""Per-problem slice report for comparing Full OPD and Oracle RA eval runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import json
import math
import multiprocessing
import os
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


@dataclass
class Rollout:
    run_name: str
    task: str
    example_id: int
    seed: int
    prompt: str
    answer: str
    response: str
    correct: bool
    response_len_tokens: int
    response_len_chars: int


def parse_tasks(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def find_task_file(eval_dir: Path, task: str, max_tokens_tag: str) -> Path:
    matches = sorted(eval_dir.glob(f"{task.lower()}_*{max_tokens_tag}.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No JSONL found for task={task} tag={max_tokens_tag} under {eval_dir}")
    if len(matches) > 1:
        exact = [path for path in matches if f"-{max_tokens_tag}.jsonl" in path.name]
        if exact:
            matches = exact
    return matches[0]


def safe_grade(response: str, answer: str) -> bool:
    try:
        return bool(grade_answer_verl(response, answer))
    except Exception as exc:
        print(f"[warn] grade failed; marking incorrect: {exc}", flush=True)
        return False


def load_run_rollouts(
    eval_dir: Path,
    run_name: str,
    tasks: list[str],
    max_tokens_tag: str,
    tokenizer: Any,
) -> list[Rollout]:
    rows: list[Rollout] = []
    for task in tasks:
        path = find_task_file(eval_dir, task, max_tokens_tag)
        print(f"[load] run={run_name} task={task} file={path}", flush=True)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                response = str(obj.get("response", ""))
                answer = str(obj.get("answer", ""))
                token_len = len(tokenizer.encode(response, add_special_tokens=False))
                rows.append(
                    Rollout(
                        run_name=run_name,
                        task=task,
                        example_id=int(obj.get("example_id", -1)),
                        seed=int(obj.get("seed", -1)),
                        prompt=str(obj.get("prompt", "")),
                        answer=answer,
                        response=response,
                        correct=safe_grade(response, answer),
                        response_len_tokens=token_len,
                        response_len_chars=len(response),
                    )
                )
    print(f"[load] run={run_name} rollouts={len(rows)}", flush=True)
    return rows


def group_rollouts(rows: list[Rollout]) -> dict[tuple[str, int], list[Rollout]]:
    grouped: dict[tuple[str, int], list[Rollout]] = defaultdict(list)
    for row in rows:
        grouped[(row.task, row.example_id)].append(row)
    return grouped


def mean(values: list[float]) -> float:
    values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return sum(values) / len(values) if values else float("nan")


def pass_value(rows: list[Rollout]) -> bool:
    return any(row.correct for row in rows)


def correct_count(rows: list[Rollout]) -> int:
    return sum(1 for row in rows if row.correct)


def slice_label(full_pass: bool, ra_pass: bool, delta_avg: float) -> str:
    if full_pass and not ra_pass:
        return "Full pass, RA fail"
    if not full_pass and ra_pass:
        return "Full fail, RA pass"
    if not full_pass and not ra_pass:
        return "both solve_none"
    if delta_avg > 0:
        return "both pass, RA avg higher"
    if delta_avg < 0:
        return "both pass, RA avg lower"
    return "both pass, RA avg equal"


def split_rollout_scores(full_rows: list[Rollout], ra_rows: list[Rollout]) -> list[dict[str, Any]]:
    output = []
    for row in full_rows + ra_rows:
        output.append(
            {
                "task": row.task,
                "example_id": row.example_id,
                "run_name": row.run_name,
                "seed": row.seed,
                "correct": row.correct,
                "response_len_tokens": row.response_len_tokens,
                "response_len_chars": row.response_len_chars,
            }
        )
    output.sort(key=lambda item: (item["task"], item["example_id"], item["run_name"], item["seed"]))
    return output


def build_gate_requests(
    rows: list[Rollout],
    tokenizer: Any,
    prefix_frac: float,
    max_tokens: int,
    max_new_tokens_cap: int,
    enable_thinking: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    immediate: list[dict[str, Any]] = []

    formatted_prompt_cache: dict[tuple[str, str], str] = {}
    for row in rows:
        response_ids = tokenizer.encode(row.response, add_special_tokens=False)
        valid_len = len(response_ids)
        base = {
            "task": row.task,
            "example_id": row.example_id,
            "run_name": row.run_name,
            "seed": row.seed,
            "correct": row.correct,
            "response_len_tokens": row.response_len_tokens,
            "response_len_chars": row.response_len_chars,
        }
        if valid_len <= 0:
            immediate.append(
                {
                    **base,
                    "rho75": float("nan"),
                    "tail75_gate": 1.0,
                    "full_response_gate_mean": 1.0,
                    "prefix_token_len": 0,
                    "tail_token_len": 0,
                    "max_new_tokens": 0,
                    "gate_replay_status": "empty_response",
                }
            )
            continue

        prefix_len = int(math.floor(valid_len * prefix_frac))
        prefix_len = max(1, min(valid_len, prefix_len))
        prefix_ids = response_ids[:prefix_len]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        tail_len = max(0, valid_len - prefix_len)
        remaining_tokens = max_tokens - prefix_len
        if remaining_tokens <= 0:
            immediate.append(
                {
                    **base,
                    "rho75": float("nan"),
                    "tail75_gate": 1.0,
                    "full_response_gate_mean": 1.0,
                    "prefix_token_len": prefix_len,
                    "tail_token_len": tail_len,
                    "max_new_tokens": 0,
                    "gate_replay_status": "prefix_exceeds_max_tokens",
                }
            )
            continue

        prompt_key = (row.task, row.prompt)
        formatted_prompt = formatted_prompt_cache.get(prompt_key)
        if formatted_prompt is None:
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row.prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            formatted_prompt_cache[prompt_key] = formatted_prompt

        max_new_tokens = min(int(max_new_tokens_cap), int(remaining_tokens))
        request_id = f"{row.run_name}:{row.task}:{row.example_id}:seed{row.seed}:f{prefix_frac:g}"
        requests.append(
            {
                **base,
                "request_id": request_id,
                "answer": row.answer,
                "prefix_text": prefix_text,
                "prompt_text": formatted_prompt + prefix_text,
                "prefix_token_len": prefix_len,
                "tail_token_len": tail_len,
                "max_new_tokens": max_new_tokens,
            }
        )

    return requests, immediate


def split_requests_by_load(requests: list[dict[str, Any]], num_workers: int, num_continuations: int) -> list[list[dict[str, Any]]]:
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
    loads = [0 for _ in range(num_workers)]
    for request in sorted(requests, key=lambda item: int(item["max_new_tokens"]) * num_continuations, reverse=True):
        idx = min(range(num_workers), key=lambda worker_idx: loads[worker_idx])
        shards[idx].append(request)
        loads[idx] += int(request["max_new_tokens"]) * num_continuations
    return shards


def worker_replay_gate(args_tuple: tuple[Any, ...]) -> list[dict[str, Any]]:
    (
        model_path,
        requests,
        gpu_id,
        num_continuations,
        temperature,
        top_p,
        gpu_memory_utilization,
        tensor_parallel_size,
        generation_batch_size,
        min_gate,
    ) = args_tuple

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from vllm import LLM, SamplingParams

    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment
    except ImportError:  # pragma: no cover
        destroy_distributed_environment = None
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
    except ImportError:  # pragma: no cover
        destroy_model_parallel = None

    llm = None
    rows: list[dict[str, Any]] = []
    stop_token_ids: list[int] = []
    try:
        print(f"[GPU {gpu_id}] loading teacher={model_path}; requests={len(requests)}", flush=True)
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=float(gpu_memory_utilization),
            tensor_parallel_size=int(tensor_parallel_size),
        )
        tokenizer = llm.get_tokenizer()
        for stop_token in ["<|im_end|>", "<|endoftext|>"]:
            try:
                encoded = tokenizer.encode(stop_token, add_special_tokens=False)
                if encoded:
                    stop_token_ids.append(int(encoded[0]))
            except Exception:
                pass

        batch_size = generation_batch_size if generation_batch_size and generation_batch_size > 0 else len(requests)
        for start in range(0, len(requests), batch_size):
            chunk = requests[start : start + batch_size]
            prompts = [request["prompt_text"] for request in chunk]
            sampling = [
                SamplingParams(
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_tokens=int(request["max_new_tokens"]),
                    n=int(num_continuations),
                    stop_token_ids=stop_token_ids if stop_token_ids else None,
                )
                for request in chunk
            ]
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            for request, output in zip(chunk, outputs, strict=True):
                correct_values = []
                for sample in output.outputs:
                    full_response = request["prefix_text"] + sample.text
                    correct_values.append(1.0 if safe_grade(full_response, str(request["answer"])) else 0.0)
                rho = mean(correct_values)
                tail_gate = float(min_gate) + (1.0 - float(min_gate)) * min(1.0, max(0.0, rho))
                valid_len = int(request["response_len_tokens"])
                tail_len = int(request["tail_token_len"])
                prefix_len = int(request["prefix_token_len"])
                full_gate = (
                    (prefix_len + tail_len * tail_gate) / valid_len if valid_len > 0 else 1.0
                )
                rows.append(
                    {
                        "task": request["task"],
                        "example_id": request["example_id"],
                        "run_name": request["run_name"],
                        "seed": request["seed"],
                        "correct": request["correct"],
                        "response_len_tokens": request["response_len_tokens"],
                        "response_len_chars": request["response_len_chars"],
                        "rho75": rho,
                        "tail75_gate": tail_gate,
                        "full_response_gate_mean": full_gate,
                        "prefix_token_len": prefix_len,
                        "tail_token_len": tail_len,
                        "max_new_tokens": request["max_new_tokens"],
                        "gate_replay_status": "ok",
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
    return rows


def replay_gates(
    rows: list[Rollout],
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    requests, immediate = build_gate_requests(
        rows=rows,
        tokenizer=tokenizer,
        prefix_frac=args.prefix_frac,
        max_tokens=args.max_tokens,
        max_new_tokens_cap=args.max_new_tokens_cap,
        enable_thinking=args.thinking_flag == "enable",
    )
    print(f"[gate] requests={len(requests)} immediate={len(immediate)}", flush=True)
    if not requests:
        return immediate

    gpu_workers = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_workers:
        raise ValueError("--gpus must contain at least one GPU id when gate replay is enabled")
    shards = split_requests_by_load(requests, len(gpu_workers), args.num_continuations)
    worker_args = [
        (
            str(args.teacher_model_path),
            shard,
            gpu_workers[idx],
            args.num_continuations,
            args.temperature,
            args.top_p,
            args.gpu_memory_utilization,
            args.tensor_parallel_size,
            args.generation_batch_size,
            args.min_gate,
        )
        for idx, shard in enumerate(shards)
        if shard
    ]

    replay_rows: list[dict[str, Any]] = list(immediate)
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=ctx) as executor:
        futures = [executor.submit(worker_replay_gate, item) for item in worker_args]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Gate replay workers"):
            replay_rows.extend(future.result())

    replay_rows.sort(key=lambda item: (item["task"], item["example_id"], item["run_name"], item["seed"]))
    return replay_rows


def aggregate_gate(rows: list[dict[str, Any]], run_name: str) -> dict[str, Any]:
    if not rows:
        return {
            f"{run_name}_run_gate_mean": float("nan"),
            f"{run_name}_run_tail75_gate_mean": float("nan"),
            f"{run_name}_correct_tail75_gate_mean": float("nan"),
            f"{run_name}_wrong_tail75_gate_mean": float("nan"),
            f"{run_name}_response_len_mean": float("nan"),
            f"{run_name}_correct_response_len_mean": float("nan"),
            f"{run_name}_wrong_response_len_mean": float("nan"),
        }
    correct_rows = [row for row in rows if bool(row["correct"])]
    wrong_rows = [row for row in rows if not bool(row["correct"])]
    return {
        f"{run_name}_run_gate_mean": mean([row["full_response_gate_mean"] for row in rows]),
        f"{run_name}_run_tail75_gate_mean": mean([row["tail75_gate"] for row in rows]),
        f"{run_name}_correct_tail75_gate_mean": mean([row["tail75_gate"] for row in correct_rows]),
        f"{run_name}_wrong_tail75_gate_mean": mean([row["tail75_gate"] for row in wrong_rows]),
        f"{run_name}_response_len_mean": mean([row["response_len_tokens"] for row in rows]),
        f"{run_name}_correct_response_len_mean": mean([row["response_len_tokens"] for row in correct_rows]),
        f"{run_name}_wrong_response_len_mean": mean([row["response_len_tokens"] for row in wrong_rows]),
    }


def build_problem_slices(
    full_rows: list[Rollout],
    ra_rows: list[Rollout],
    gate_rows: list[dict[str, Any]],
    expected_n: int,
) -> list[dict[str, Any]]:
    full_grouped = group_rollouts(full_rows)
    ra_grouped = group_rollouts(ra_rows)
    common_keys = sorted(set(full_grouped).intersection(ra_grouped))
    gate_grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gate_rows:
        gate_grouped[(row["task"], int(row["example_id"]), row["run_name"])].append(row)

    problem_rows = []
    for task, example_id in common_keys:
        full_problem_rows = sorted(full_grouped[(task, example_id)], key=lambda row: row.seed)
        ra_problem_rows = sorted(ra_grouped[(task, example_id)], key=lambda row: row.seed)
        full_correct_count = correct_count(full_problem_rows)
        ra_correct_count = correct_count(ra_problem_rows)
        full_avg = full_correct_count / len(full_problem_rows) if full_problem_rows else float("nan")
        ra_avg = ra_correct_count / len(ra_problem_rows) if ra_problem_rows else float("nan")
        full_pass = full_correct_count > 0
        ra_pass = ra_correct_count > 0
        delta_avg = ra_avg - full_avg
        full_gate_rows = gate_grouped[(task, example_id, "full")]
        ra_gate_rows = gate_grouped[(task, example_id, "ra")]
        row = {
            "task": task,
            "example_id": example_id,
            "slice": slice_label(full_pass, ra_pass, delta_avg),
            "full_rollouts": len(full_problem_rows),
            "ra_rollouts": len(ra_problem_rows),
            "expected_n": expected_n,
            "full_avg": full_avg,
            "full_pass": full_pass,
            "full_correct_count": full_correct_count,
            "ra_avg": ra_avg,
            "ra_pass": ra_pass,
            "ra_correct_count": ra_correct_count,
            "delta_avg": delta_avg,
            **aggregate_gate(full_gate_rows, "full"),
            **aggregate_gate(ra_gate_rows, "ra"),
        }
        problem_rows.append(row)
    return problem_rows


def summarize_slices(problem_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in problem_rows:
        grouped[str(row["slice"])].append(row)
    summary = []
    for label, rows in sorted(grouped.items()):
        summary.append(
            {
                "slice": label,
                "num_problems": len(rows),
                "full_avg_mean": mean([row["full_avg"] for row in rows]),
                "ra_avg_mean": mean([row["ra_avg"] for row in rows]),
                "delta_avg_mean": mean([row["delta_avg"] for row in rows]),
                "full_correct_tail75_gate_mean": mean([row["full_correct_tail75_gate_mean"] for row in rows]),
                "full_wrong_tail75_gate_mean": mean([row["full_wrong_tail75_gate_mean"] for row in rows]),
                "ra_correct_tail75_gate_mean": mean([row["ra_correct_tail75_gate_mean"] for row in rows]),
                "ra_wrong_tail75_gate_mean": mean([row["ra_wrong_tail75_gate_mean"] for row in rows]),
                "full_len_mean": mean([row["full_response_len_mean"] for row in rows]),
                "ra_len_mean": mean([row["ra_response_len_mean"] for row in rows]),
            }
        )
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.4f}"


def write_full_pass_ra_fail(path: Path, problem_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in problem_rows if row["slice"] == "Full pass, RA fail"]
    lines = ["# Full Pass, RA Fail Problems", ""]
    lines.append(
        "| task | example_id | full avg/pass/count | RA avg/pass/count | full correct tail gate | full wrong tail gate | RA wrong tail gate | full len | RA len |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {task} | {eid} | {fa}/{fp}/{fc} | {ra}/{rp}/{rc} | {fgc} | {fgw} | {rgw} | {flen} | {rlen} |".format(
                task=row["task"],
                eid=row["example_id"],
                fa=fmt(row["full_avg"]),
                fp=fmt(row["full_pass"]),
                fc=row["full_correct_count"],
                ra=fmt(row["ra_avg"]),
                rp=fmt(row["ra_pass"]),
                rc=row["ra_correct_count"],
                fgc=fmt(row["full_correct_tail75_gate_mean"]),
                fgw=fmt(row["full_wrong_tail75_gate_mean"]),
                rgw=fmt(row["ra_wrong_tail75_gate_mean"]),
                flen=fmt(row["full_response_len_mean"]),
                rlen=fmt(row["ra_response_len_mean"]),
            )
        )
    if not rows:
        lines.append("")
        lines.append("No Full pass, RA fail problems found.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(path: Path, problem_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    lines = ["# Full OPD vs Oracle RA Tail75 Slice Report", ""]
    lines.append("## Overall")
    lines.append("")
    full_pass = mean([1.0 if row["full_pass"] else 0.0 for row in problem_rows])
    ra_pass = mean([1.0 if row["ra_pass"] else 0.0 for row in problem_rows])
    full_avg = mean([row["full_avg"] for row in problem_rows])
    ra_avg = mean([row["ra_avg"] for row in problem_rows])
    lines.append(f"- Full OPD avg/pass: {fmt(full_avg)} / {fmt(full_pass)}")
    lines.append(f"- Oracle RA tail75 avg/pass: {fmt(ra_avg)} / {fmt(ra_pass)}")
    lines.append(f"- Delta avg/pass (RA - Full): {fmt(ra_avg - full_avg)} / {fmt(ra_pass - full_pass)}")
    lines.append("")

    lines.append("## Slice Summary")
    lines.append("")
    lines.append(
        "| slice | problems | Full avg | RA avg | delta avg | Full correct gate | Full wrong gate | RA correct gate | RA wrong gate | Full len | RA len |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {slice} | {n} | {fa} | {ra} | {delta} | {fcg} | {fwg} | {rcg} | {rwg} | {flen} | {rlen} |".format(
                slice=row["slice"],
                n=row["num_problems"],
                fa=fmt(row["full_avg_mean"]),
                ra=fmt(row["ra_avg_mean"]),
                delta=fmt(row["delta_avg_mean"]),
                fcg=fmt(row["full_correct_tail75_gate_mean"]),
                fwg=fmt(row["full_wrong_tail75_gate_mean"]),
                rcg=fmt(row["ra_correct_tail75_gate_mean"]),
                rwg=fmt(row["ra_wrong_tail75_gate_mean"]),
                flen=fmt(row["full_len_mean"]),
                rlen=fmt(row["ra_len_mean"]),
            )
        )
    lines.append("")

    target = [row for row in summary_rows if row["slice"] == "Full pass, RA fail"]
    lines.append("## Diagnostic Notes")
    lines.append("")
    if target:
        row = target[0]
        full_correct_gate = row["full_correct_tail75_gate_mean"]
        full_wrong_gate = row["full_wrong_tail75_gate_mean"]
        ra_wrong_gate = row["ra_wrong_tail75_gate_mean"]
        lines.append(
            f"- Full pass, RA fail problems: {row['num_problems']}; "
            f"Full correct tail gate={fmt(full_correct_gate)}, "
            f"Full wrong tail gate={fmt(full_wrong_gate)}, RA wrong tail gate={fmt(ra_wrong_gate)}."
        )
        if not math.isnan(float(full_correct_gate)) and full_correct_gate < 0.5:
            lines.append("- Full correct rollouts also receive low tail gates; this supports the gate false-positive / recall-too-high concern.")
        elif not math.isnan(float(full_correct_gate)) and full_correct_gate >= 0.7:
            lines.append("- Full correct rollouts keep high tail gates; RA loss is less likely to be a simple counterfactual gate false positive on those paths.")
        else:
            lines.append("- Full correct rollout gates are intermediate; inspect `full_pass_ra_fail.md` problem by problem.")
    else:
        lines.append("- No Full pass, RA fail problems were found in this slice.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_common_problem_subset(
    full_rows: list[Rollout],
    ra_rows: list[Rollout],
    max_problems: int | None,
) -> tuple[list[Rollout], list[Rollout]]:
    if max_problems is None:
        return full_rows, ra_rows
    full_keys = set(group_rollouts(full_rows))
    ra_keys = set(group_rollouts(ra_rows))
    selected = set(sorted(full_keys.intersection(ra_keys))[:max_problems])
    full_selected = [row for row in full_rows if (row.task, row.example_id) in selected]
    ra_selected = [row for row in ra_rows if (row.task, row.example_id) in selected]
    print(f"[debug] selected max_problems={max_problems}; full={len(full_selected)} ra={len(ra_selected)}", flush=True)
    return full_selected, ra_selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Full OPD and Oracle RA runs by per-problem slices.")
    parser.add_argument("--full-eval-dir", required=True)
    parser.add_argument("--ra-eval-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tasks", default="aime25")
    parser.add_argument("--max-tokens-tag", default="MNT31744")
    parser.add_argument("--teacher-model-path", default=str(ROOT_DIR / "model" / "JustRL-DeepSeek-1.5B"))
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer for response length and prefix construction.")
    parser.add_argument("--prefix-frac", type=float, default=0.75)
    parser.add_argument("--num-continuations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=31744)
    parser.add_argument("--max-new-tokens-cap", type=int, default=1024)
    parser.add_argument("--min-gate", type=float, default=0.1)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--thinking-flag", choices=["disable", "enable"], default="disable")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--expected-n", type=int, default=16)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--skip-gate-replay", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = parse_tasks(args.tasks)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.replace:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = args.tokenizer_path or args.teacher_model_path
    print(f"[tokenizer] loading {tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True, local_files_only=True)

    full_rows = load_run_rollouts(Path(args.full_eval_dir), "full", tasks, args.max_tokens_tag, tokenizer)
    ra_rows = load_run_rollouts(Path(args.ra_eval_dir), "ra", tasks, args.max_tokens_tag, tokenizer)
    full_rows, ra_rows = select_common_problem_subset(full_rows, ra_rows, args.max_problems)

    config = vars(args).copy()
    config["tasks"] = tasks
    config["full_eval_dir"] = str(Path(args.full_eval_dir))
    config["ra_eval_dir"] = str(Path(args.ra_eval_dir))
    config["tokenizer_path"] = str(tokenizer_path)
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rollout_scores = split_rollout_scores(full_rows, ra_rows)
    write_jsonl(out_dir / "rollout_scores.jsonl", rollout_scores)

    if args.skip_gate_replay:
        gate_rows = []
    else:
        gate_rows = replay_gates(full_rows + ra_rows, tokenizer, args)
    write_jsonl(out_dir / "rollout_gate_stats.jsonl", gate_rows)

    problem_rows = build_problem_slices(full_rows, ra_rows, gate_rows, args.expected_n)
    summary_rows = summarize_slices(problem_rows)
    write_csv(out_dir / "problem_slices.csv", problem_rows)
    write_csv(out_dir / "slice_summary.csv", summary_rows)
    write_full_pass_ra_fail(out_dir / "full_pass_ra_fail.md", problem_rows)
    write_summary_md(out_dir / "summary.md", problem_rows, summary_rows)

    print(f"[done] report written to {out_dir}", flush=True)
    print(f"[done] summary: {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
