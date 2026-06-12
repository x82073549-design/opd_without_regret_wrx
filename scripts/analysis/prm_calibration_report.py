#!/usr/bin/env python3
"""Build a PRM calibration report from existing rollout JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_UTILS_DIR = ROOT_DIR / "scripts" / "val" / "eval"
if str(EVAL_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_UTILS_DIR))

from utils import grade_answer_verl  # noqa: E402


CORE_FEATURES = [
    "V_last",
    "V_mean",
    "V_min",
    "V_softmin",
    "drop_min",
    "drop_sum_neg",
]

SCORE_LIKE_FEATURES = ["V_last", "V_mean", "V_min", "V_softmin"]

ALL_FEATURES = CORE_FEATURES + [
    "first_low_pos_norm",
    "gate_v1_mean",
    "gate_v1_low_frac",
    "gate_v2_mean",
    "gate_v2_std",
    "gate_v2_gt1_frac",
    "gate_v2_low_frac",
    "gate_v2_oracle_mean",
    "gate_v2_oracle_std",
    "gate_v2_oracle_gt1_frac",
    "gate_v2_oracle_low_frac",
]


@dataclass
class RunningStats:
    momentum: float = 0.95
    mean: float | None = None
    sq_mean: float | None = None
    count: int = 0

    def current(self) -> tuple[float | None, float | None, int]:
        if self.mean is None or self.sq_mean is None:
            return None, None, self.count
        variance = max(0.0, self.sq_mean - self.mean * self.mean)
        return self.mean, math.sqrt(variance), self.count

    def update(self, values: list[float]) -> None:
        if not values:
            return
        batch_mean = sum(values) / len(values)
        batch_sq_mean = sum(value * value for value in values) / len(values)
        if self.mean is None or self.sq_mean is None:
            self.mean = batch_mean
            self.sq_mean = batch_sq_mean
        else:
            self.mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
            self.sq_mean = self.momentum * self.sq_mean + (1.0 - self.momentum) * batch_sq_mean
        self.count += len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a PRM calibration report from rollout JSONL files.")
    parser.add_argument("--eval-dir", required=True, help="Directory containing rollout jsonl files.")
    parser.add_argument(
        "--prm-model-path",
        default=str(ROOT_DIR / "model" / "Skywork-o1-Open-PRM-Qwen-2.5-1.5B"),
        help="Skywork PRM model path.",
    )
    parser.add_argument(
        "--actor-tokenizer-path",
        default=str(ROOT_DIR / "model" / "DeepSeek-R1-Distill-Qwen-1.5B"),
        help="Actor tokenizer path used to reproduce RG-OPD block splitting.",
    )
    parser.add_argument("--out-dir", required=True, help="Output report directory.")
    parser.add_argument("--tasks", default="amc23,aime24,aime25", help="Comma-separated task names.")
    parser.add_argument("--max-tokens-tag", default="MNT31744", help="Filename max-tokens tag to select.")
    parser.add_argument("--delimiter", default="\n\n", help="Delimiter used for raw response blocks.")
    parser.add_argument("--n-prm-blocks", type=int, default=64, help="Maximum grouped PRM blocks per rollout.")
    parser.add_argument("--prm-step-token", default="\n", help="Step marker appended to each PRM block.")
    parser.add_argument("--tau", type=float, default=0.1, help="Softmin temperature.")
    parser.add_argument("--low-threshold", type=float, default=0.5, help="Threshold for first_low_pos.")
    parser.add_argument("--device", default="cuda", help="Device for PRM inference.")
    parser.add_argument("--micro-batch-size", type=int, default=8, help="PRM inference micro batch size.")
    parser.add_argument("--max-rollouts-per-task", type=int, default=None, help="Optional debug limit per task.")
    parser.add_argument("--curve-bins", type=int, default=16, help="Number of normalized block-position bins.")
    parser.add_argument("--ece-bins", type=int, default=10, help="Number of ECE bins.")
    parser.add_argument("--ema-lambda", type=float, default=0.6, help="EMA lambda for simulated gates.")
    parser.add_argument("--v1-min-gate", type=float, default=0.05, help="Minimum gate for simulated RG-v1.")
    parser.add_argument("--v1-low-gate-threshold", type=float, default=0.5, help="Low gate threshold for RG-v1.")
    parser.add_argument("--v2-min-gate", type=float, default=0.5, help="Minimum gate for simulated RG-v2.")
    parser.add_argument("--v2-max-gate", type=float, default=1.5, help="Maximum gate for simulated RG-v2.")
    parser.add_argument("--v2-gate-alpha", type=float, default=0.25, help="Gate alpha for simulated RG-v2.")
    parser.add_argument("--v2-std-floor", type=float, default=0.05, help="Std floor for simulated RG-v2.")
    return parser.parse_args()


def normalize_tasks(tasks: str) -> list[str]:
    return [task.strip().lower() for task in tasks.split(",") if task.strip()]


def find_task_file(eval_dir: Path, task: str, max_tokens_tag: str) -> Path:
    candidates = sorted(eval_dir.glob(f"{task}_*.jsonl"))
    candidates = [path for path in candidates if max_tokens_tag in path.name]
    if not candidates:
        raise FileNotFoundError(f"No jsonl found for task={task} max_tokens_tag={max_tokens_tag} under {eval_dir}")
    if len(candidates) > 1:
        print(f"[warn] multiple files for task={task}; using {candidates[0]}", flush=True)
    return candidates[0]


def load_rollouts(eval_dir: Path, tasks: list[str], max_tokens_tag: str, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        file_path = find_task_file(eval_dir, task, max_tokens_tag)
        count = 0
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and count >= limit:
                    break
                data = json.loads(line)
                rows.append(
                    {
                        "task": task,
                        "source_file": str(file_path),
                        "example_id": int(data.get("example_id", -1)),
                        "seed": data.get("seed"),
                        "prompt": str(data.get("prompt", "")),
                        "answer": str(data.get("answer", "")),
                        "response": str(data.get("response", "")),
                    }
                )
                count += 1
        print(f"[load] task={task} rollouts={count} file={file_path}", flush=True)
    return rows


def safe_grade(response: str, answer: str) -> bool:
    try:
        return bool(grade_answer_verl(response, answer))
    except Exception as exc:
        print(f"[warn] grade failed; marking incorrect: {exc}", flush=True)
        return False


def delimiter_token_candidates(actor_tokenizer: Any, delimiter: str) -> list[tuple[int, ...]]:
    candidates: list[tuple[int, ...]] = []

    def add_candidate(token_ids: Iterable[int]) -> None:
        candidate = tuple(int(token_id) for token_id in token_ids)
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add_candidate(actor_tokenizer.encode(delimiter, add_special_tokens=False))
    if delimiter == "\n\n":
        newline_ids = actor_tokenizer.encode("\n", add_special_tokens=False)
        add_candidate(newline_ids + newline_ids)
    candidates.sort(key=len, reverse=True)
    return candidates


def decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def split_delimiter_grouped_token_spans(
    token_ids: list[int],
    actor_tokenizer: Any,
    delimiter: str,
    n_prm_blocks: int,
    piece_cache: dict[int, str],
) -> list[tuple[int, int, str]]:
    if not token_ids:
        return []

    candidates = delimiter_token_candidates(actor_tokenizer, delimiter)
    raw_spans: list[tuple[int, int]] = []
    block_start = 0
    pos = 0
    while pos < len(token_ids):
        matched_len = 0
        for candidate in candidates:
            cand_len = len(candidate)
            if cand_len and tuple(token_ids[pos : pos + cand_len]) == candidate:
                matched_len = cand_len
                break
        if matched_len == 0 and delimiter:
            token_id = int(token_ids[pos])
            piece = piece_cache.get(token_id)
            if piece is None:
                piece = decode_ids(actor_tokenizer, [token_id])
                piece_cache[token_id] = piece
            if delimiter in piece:
                matched_len = 1

        if matched_len > 0:
            block_end = min(len(token_ids), pos + matched_len)
            block_text = decode_ids(actor_tokenizer, token_ids[block_start:block_end])
            if block_text.strip():
                raw_spans.append((block_start, block_end))
            block_start = block_end
            pos = block_end
        else:
            pos += 1

    if block_start < len(token_ids):
        block_text = decode_ids(actor_tokenizer, token_ids[block_start:])
        if block_text.strip():
            raw_spans.append((block_start, len(token_ids)))

    if not raw_spans:
        raw_spans = [(0, len(token_ids))]

    n_prm_blocks = max(1, int(n_prm_blocks))
    raw_block_num = len(raw_spans)
    group_size = max(1, (raw_block_num + n_prm_blocks - 1) // n_prm_blocks)
    grouped: list[tuple[int, int, str]] = []
    for start in range(0, raw_block_num, group_size):
        end = min(raw_block_num, start + group_size)
        token_start = raw_spans[start][0]
        token_end = raw_spans[end - 1][1]
        block_text = decode_ids(actor_tokenizer, token_ids[token_start:token_end])
        if token_end > token_start and block_text.strip():
            grouped.append((token_start, token_end, block_text))
    return grouped


def build_prm_input(prm_tokenizer: Any, prompt: str, block_texts: list[str], step_token: str) -> tuple[list[int], list[int]]:
    input_ids = prm_tokenizer.encode(prompt + "\n\n", add_special_tokens=False)
    reward_flags: list[int] = []
    for block_text in block_texts:
        step_text = block_text.strip()
        if not step_text:
            continue
        step_ids = prm_tokenizer.encode(step_text + step_token, add_special_tokens=False)
        if not step_ids:
            continue
        input_ids.extend(step_ids)
        reward_flags.append(len(input_ids) - 1)
    if not input_ids:
        fallback_id = prm_tokenizer.eos_token_id if prm_tokenizer.eos_token_id is not None else prm_tokenizer.pad_token_id
        input_ids = [fallback_id if fallback_id is not None else 0]
    return input_ids, reward_flags


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(device_name)


def load_prm(prm_model_path: str, device: torch.device) -> tuple[Any, Any, torch.dtype]:
    load_start = time.time()
    model_path = Path(prm_model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    config.torch_dtype = dtype
    model = AutoModel.from_config(config, trust_remote_code=True)

    weight_path = model_path / "model.safetensors"
    if weight_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(f"{weight_path} exists but safetensors is not installed") from exc
        state_dict = load_file(str(weight_path), device="cpu")
    else:
        weight_path = model_path / "pytorch_model.bin"
        if not weight_path.exists():
            raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found under {model_path}")
        state_dict = torch.load(str(weight_path), map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(
            f"[warn] PRM load_state_dict missing={len(missing_keys)} unexpected={len(unexpected_keys)}",
            flush=True,
        )
    model = model.to(device=device, dtype=dtype)
    model.eval()
    print(f"[load] PRM loaded from {weight_path} in {time.time() - load_start:.1f}s", flush=True)
    return tokenizer, model, dtype


def score_prm_examples(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    dtype: torch.dtype,
    micro_batch_size: int,
) -> list[list[float]]:
    if not examples:
        return []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    all_scores: list[list[float]] = []
    micro_batch_size = max(1, int(micro_batch_size))
    for start in range(0, len(examples), micro_batch_size):
        chunk = examples[start : start + micro_batch_size]
        max_len = max(len(example["input_ids"]) for example in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        for row, example in enumerate(chunk):
            ids = torch.tensor(example["input_ids"], dtype=torch.long, device=device)
            input_ids[row, : ids.numel()] = ids
            attention_mask[row, : ids.numel()] = 1

        autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        with torch.no_grad():
            if autocast_enabled:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    transformer_outputs = model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    logits = model.v_head(transformer_outputs.last_hidden_state).squeeze(-1)
            else:
                transformer_outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                logits = model.v_head(transformer_outputs.last_hidden_state).squeeze(-1)
            probs = torch.sigmoid(logits.float())

        for row, example in enumerate(chunk):
            flags = example["reward_flags"]
            if flags:
                flag_tensor = torch.tensor(flags, dtype=torch.long, device=device)
                scores = probs[row].index_select(0, flag_tensor).detach().cpu().tolist()
            else:
                scores = []
            all_scores.append([float(score) for score in scores])
        print(f"[prm] scored {min(start + len(chunk), len(examples))}/{len(examples)}", flush=True)
    return all_scores


def softmin(values: list[float], tau: float) -> float:
    if not values:
        return 0.0
    tau = max(float(tau), 1e-8)
    scaled = [-value / tau for value in values]
    max_scaled = max(scaled)
    log_mean = max_scaled + math.log(sum(math.exp(value - max_scaled) for value in scaled) / len(values))
    return -tau * log_mean


def simulate_v1_gate(values: list[float], ema_lambda: float, min_gate: float, low_gate_threshold: float) -> dict[str, float]:
    if not values:
        return {"gate_v1_mean": 1.0, "gate_v1_low_frac": 0.0}
    gates: list[float] = []
    prev_gate = 1.0
    for value in values:
        gate = max(min_gate, ema_lambda * prev_gate + (1.0 - ema_lambda) * value)
        gates.append(gate)
        prev_gate = gate
    return {
        "gate_v1_mean": mean(gates),
        "gate_v1_low_frac": sum(1 for gate in gates if gate < low_gate_threshold) / len(gates),
    }


def simulate_v2_gate(
    values: list[float],
    running_mean: float | None,
    running_std: float | None,
    running_count: int,
    ema_lambda: float,
    min_gate: float,
    max_gate: float,
    gate_alpha: float,
    std_floor: float,
) -> dict[str, float]:
    if not values:
        return {
            "mean": 1.0,
            "std": 0.0,
            "gt1_frac": 0.0,
            "low_frac": 0.0,
        }
    stats_ready = running_mean is not None and running_std is not None and running_count > 0
    denom = max(running_std if running_std is not None else 0.0, std_floor)
    gates: list[float] = []
    prev_gate = 1.0
    for value in values:
        if stats_ready:
            z_value = (value - float(running_mean)) / denom
            raw_gate = max(min_gate, min(max_gate, 1.0 + gate_alpha * z_value))
            gate = ema_lambda * prev_gate + (1.0 - ema_lambda) * raw_gate
        else:
            gate = 1.0
        gates.append(gate)
        prev_gate = gate
    return {
        "mean": mean(gates),
        "std": std(gates),
        "gt1_frac": sum(1 for gate in gates if gate > 1.0) / len(gates),
        "low_frac": sum(1 for gate in gates if gate < 1.0) / len(gates),
    }


def compute_features(values: list[float], args: argparse.Namespace) -> dict[str, Any]:
    if not values:
        values = [0.0]
    deltas = [values[idx] - values[idx - 1] for idx in range(1, len(values))]
    if deltas:
        drop_min = min(deltas)
        drop_sum_neg = sum(min(0.0, delta) for delta in deltas)
    else:
        drop_min = 0.0
        drop_sum_neg = 0.0

    first_low_pos = -1
    for idx, value in enumerate(values):
        if value < args.low_threshold:
            first_low_pos = idx
            break
    first_low_pos_norm = -1.0 if first_low_pos < 0 else (first_low_pos + 1) / len(values)

    features: dict[str, Any] = {
        "V_last": values[-1],
        "V_mean": mean(values),
        "V_min": min(values),
        "V_softmin": softmin(values, args.tau),
        "drop_min": drop_min,
        "drop_sum_neg": drop_sum_neg,
        "first_low_pos": first_low_pos,
        "first_low_pos_norm": first_low_pos_norm,
    }
    features.update(simulate_v1_gate(values, args.ema_lambda, args.v1_min_gate, args.v1_low_gate_threshold))
    return features


def add_v2_features(
    record: dict[str, Any],
    values: list[float],
    stats: tuple[float | None, float | None, int],
    args: argparse.Namespace,
    prefix: str = "gate_v2",
) -> None:
    gate = simulate_v2_gate(
        values,
        running_mean=stats[0],
        running_std=stats[1],
        running_count=stats[2],
        ema_lambda=args.ema_lambda,
        min_gate=args.v2_min_gate,
        max_gate=args.v2_max_gate,
        gate_alpha=args.v2_gate_alpha,
        std_floor=args.v2_std_floor,
    )
    record[f"{prefix}_mean"] = gate["mean"]
    record[f"{prefix}_std"] = gate["std"]
    record[f"{prefix}_gt1_frac"] = gate["gt1_frac"]
    record[f"{prefix}_low_frac"] = gate["low_frac"]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    value_mean = mean(values)
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / len(values))


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for pos in range(idx, end):
            ranks[indexed[pos][0]] = avg_rank
        idx = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_centered = [value - x_mean for value in xs]
    y_centered = [value - y_mean for value in ys]
    denom_x = math.sqrt(sum(value * value for value in x_centered))
    denom_y = math.sqrt(sum(value * value for value in y_centered))
    denom = denom_x * denom_y
    if denom == 0:
        return float("nan")
    return sum(x * y for x, y in zip(x_centered, y_centered)) / denom


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    return pearson(rankdata(xs), rankdata(ys))


def auc_score(values: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(values)
    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def summarize_feature(records: list[dict[str, Any]], group: str, feature: str) -> dict[str, Any]:
    values = [float(record[feature]) for record in records if is_finite_number(record.get(feature))]
    labels = [int(record["correct"]) for record in records if is_finite_number(record.get(feature))]
    if not values:
        return {
            "group": group,
            "feature": feature,
            "count": 0,
            "auc": float("nan"),
            "auc_oriented": float("nan"),
            "spearman": float("nan"),
            "correct_mean": float("nan"),
            "wrong_mean": float("nan"),
            "correct_std": float("nan"),
            "wrong_std": float("nan"),
            "mean_gap": float("nan"),
        }
    auc = auc_score(values, labels)
    auc_oriented = max(auc, 1.0 - auc) if math.isfinite(auc) else float("nan")
    correct_values = [value for value, label in zip(values, labels) if label == 1]
    wrong_values = [value for value, label in zip(values, labels) if label == 0]
    correct_mean = mean(correct_values) if correct_values else float("nan")
    wrong_mean = mean(wrong_values) if wrong_values else float("nan")
    return {
        "group": group,
        "feature": feature,
        "count": len(values),
        "auc": auc,
        "auc_oriented": auc_oriented,
        "spearman": spearman(values, labels),
        "correct_mean": correct_mean,
        "wrong_mean": wrong_mean,
        "correct_std": std(correct_values) if correct_values else float("nan"),
        "wrong_std": std(wrong_values) if wrong_values else float("nan"),
        "mean_gap": correct_mean - wrong_mean if math.isfinite(correct_mean) and math.isfinite(wrong_mean) else float("nan"),
    }


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def group_records(records: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "overall":
        return records
    return [record for record in records if record["task"] == group]


def compute_summary(records: list[dict[str, Any]], tasks: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in ["overall"] + tasks:
        group_rows = group_records(records, group)
        for feature in ALL_FEATURES:
            rows.append(summarize_feature(group_rows, group, feature))
    return rows


def compute_ece(
    records: list[dict[str, Any]],
    tasks: list[str],
    features: list[str],
    n_bins: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    rows: list[dict[str, Any]] = []
    ece_by_key: dict[tuple[str, str], float] = {}
    n_bins = max(1, int(n_bins))
    for group in ["overall"] + tasks:
        group_rows = group_records(records, group)
        for feature in features:
            valid = [
                (min(1.0, max(0.0, float(record[feature]))), int(record["correct"]))
                for record in group_rows
                if is_finite_number(record.get(feature))
            ]
            total = len(valid)
            ece = 0.0
            for bin_idx in range(n_bins):
                lower = bin_idx / n_bins
                upper = (bin_idx + 1) / n_bins
                if bin_idx == n_bins - 1:
                    bucket = [(conf, label) for conf, label in valid if lower <= conf <= upper]
                else:
                    bucket = [(conf, label) for conf, label in valid if lower <= conf < upper]
                count = len(bucket)
                avg_conf = mean([conf for conf, _ in bucket]) if bucket else float("nan")
                accuracy = mean([float(label) for _, label in bucket]) if bucket else float("nan")
                abs_gap = abs(avg_conf - accuracy) if math.isfinite(avg_conf) and math.isfinite(accuracy) else float("nan")
                if count and math.isfinite(abs_gap):
                    ece += (count / total) * abs_gap if total else 0.0
                rows.append(
                    {
                        "group": group,
                        "feature": feature,
                        "bin": bin_idx,
                        "lower": lower,
                        "upper": upper,
                        "count": count,
                        "avg_conf": avg_conf,
                        "accuracy": accuracy,
                        "abs_gap": abs_gap,
                    }
                )
            ece_by_key[(group, feature)] = ece if total else float("nan")
    return rows, ece_by_key


def compute_block_curves(records: list[dict[str, Any]], tasks: list[str], n_bins: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_bins = max(1, int(n_bins))
    for group in ["overall"] + tasks:
        group_rows = group_records(records, group)
        buckets: list[dict[str, list[float]]] = [{"correct": [], "wrong": []} for _ in range(n_bins)]
        for record in group_rows:
            scores = record.get("prm_scores", [])
            if not scores:
                continue
            label_key = "correct" if record["correct"] else "wrong"
            block_count = len(scores)
            for idx, value in enumerate(scores):
                bin_idx = min(n_bins - 1, int(((idx + 0.5) / block_count) * n_bins))
                buckets[bin_idx][label_key].append(float(value))
        for bin_idx, bucket in enumerate(buckets):
            correct_values = bucket["correct"]
            wrong_values = bucket["wrong"]
            correct_mean = mean(correct_values) if correct_values else float("nan")
            wrong_mean = mean(wrong_values) if wrong_values else float("nan")
            rows.append(
                {
                    "group": group,
                    "pos_bin": bin_idx,
                    "pos_lower": bin_idx / n_bins,
                    "pos_upper": (bin_idx + 1) / n_bins,
                    "correct_mean_V": correct_mean,
                    "wrong_mean_V": wrong_mean,
                    "gap": correct_mean - wrong_mean
                    if math.isfinite(correct_mean) and math.isfinite(wrong_mean)
                    else float("nan"),
                    "correct_count": len(correct_values),
                    "wrong_count": len(wrong_values),
                    "count": len(correct_values) + len(wrong_values),
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.{digits}f}"


def index_summary(summary_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["group"], row["feature"]): row for row in summary_rows}


def max_auc(summary_index: dict[tuple[str, str], dict[str, Any]], group: str, features: list[str]) -> float:
    values = [
        float(summary_index[(group, feature)]["auc_oriented"])
        for feature in features
        if (group, feature) in summary_index and is_finite_number(summary_index[(group, feature)].get("auc_oriented"))
    ]
    return max(values) if values else float("nan")


def make_stop_go_notes(summary_rows: list[dict[str, Any]], tasks: list[str]) -> list[str]:
    summary_index = index_summary(summary_rows)
    notes: list[str] = []
    aime_group = "aime25" if "aime25" in tasks else None
    if aime_group is not None:
        aime_max = max_auc(summary_index, aime_group, CORE_FEATURES)
        if math.isfinite(aime_max) and aime_max < 0.6:
            notes.append(
                "- AIME25 上 core PRM features 的 oriented AUC 都低于 0.6：RG 不应作为主线，只适合 very weak veto。"
            )
        v_last = summary_index.get((aime_group, "V_last"), {}).get("auc_oriented", float("nan"))
        other_max = max_auc(summary_index, aime_group, [feature for feature in CORE_FEATURES if feature != "V_last"])
        if is_finite_number(v_last) and float(v_last) >= 0.6 and (not math.isfinite(other_max) or other_max < 0.6):
            notes.append("- AIME25 上只有 V_last 强，block/delta features 弱：PRM 更像 ORM。")
        veto_max = max_auc(summary_index, aime_group, ["V_min", "V_softmin", "drop_min"])
        if math.isfinite(veto_max) and veto_max >= 0.6:
            notes.append("- AIME25 上 V_min / V_softmin / drop_min 至少一个较强：PRM-veto 或 dynamic truncation 有诊断依据。")
        gt1_row = summary_index.get((aime_group, "gate_v2_gt1_frac"))
        if gt1_row is not None:
            wrong_mean = gt1_row.get("wrong_mean", float("nan"))
            correct_mean = gt1_row.get("correct_mean", float("nan"))
            if is_finite_number(wrong_mean) and is_finite_number(correct_mean) and float(wrong_mean) > float(correct_mean):
                notes.append("- AIME25 上 wrong rollout 的 gate_v2_gt1_frac 更高：positive gate 风险成立。")
    if "amc23" in tasks and "aime25" in tasks:
        amc_max = max_auc(summary_index, "amc23", CORE_FEATURES)
        aime_max = max_auc(summary_index, "aime25", CORE_FEATURES)
        if math.isfinite(amc_max) and math.isfinite(aime_max) and amc_max >= 0.6 and aime_max < 0.6:
            notes.append("- AMC23 的 PRM 相关性强但 AIME25 弱：可解释 RG 对 AMC 有帮助但 hard benchmark 不稳。")
    if not notes:
        notes.append("- 未触发明确 stop/go 规则；请结合 summary.csv、calibration_bins.csv 和 block_curves.csv 继续判断。")
    return notes


def write_summary_md(path: Path, records: list[dict[str, Any]], summary_rows: list[dict[str, Any]], tasks: list[str]) -> None:
    summary_index = index_summary(summary_rows)
    lines = [
        "# PRM Calibration Report",
        "",
        "## Rollouts",
        "",
        "| group | rollouts | correct_ratio |",
        "|---|---:|---:|",
    ]
    for group in ["overall"] + tasks:
        group_rows = group_records(records, group)
        correct_ratio = mean([float(record["correct"]) for record in group_rows]) if group_rows else float("nan")
        lines.append(f"| {group} | {len(group_rows)} | {format_float(correct_ratio)} |")

    lines.extend(
        [
            "",
            "## Core Feature AUC",
            "",
            "| group | feature | auc | auc_oriented | spearman | correct_mean | wrong_mean |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group in ["overall"] + tasks:
        for feature in CORE_FEATURES:
            row = summary_index.get((group, feature), {})
            lines.append(
                "| {group} | {feature} | {auc} | {auc_o} | {sp} | {cm} | {wm} |".format(
                    group=group,
                    feature=feature,
                    auc=format_float(row.get("auc")),
                    auc_o=format_float(row.get("auc_oriented")),
                    sp=format_float(row.get("spearman")),
                    cm=format_float(row.get("correct_mean")),
                    wm=format_float(row.get("wrong_mean")),
                )
            )

    lines.extend(["", "## Gate Diagnostics", "", "| group | feature | correct_mean | wrong_mean | mean_gap |", "|---|---|---:|---:|---:|"])
    for group in ["overall"] + tasks:
        for feature in ["gate_v1_mean", "gate_v1_low_frac", "gate_v2_mean", "gate_v2_gt1_frac", "gate_v2_low_frac"]:
            row = summary_index.get((group, feature), {})
            lines.append(
                "| {group} | {feature} | {cm} | {wm} | {gap} |".format(
                    group=group,
                    feature=feature,
                    cm=format_float(row.get("correct_mean")),
                    wm=format_float(row.get("wrong_mean")),
                    gap=format_float(row.get("mean_gap")),
                )
            )

    lines.extend(["", "## Stop/Go Notes", ""])
    lines.extend(make_stop_go_notes(summary_rows, tasks))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = normalize_tasks(args.tasks)

    config = vars(args).copy()
    config["eval_dir"] = str(eval_dir)
    config["out_dir"] = str(out_dir)
    config["tasks"] = tasks
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = load_rollouts(eval_dir, tasks, args.max_tokens_tag, args.max_rollouts_per_task)
    if not rows:
        raise RuntimeError("No rollouts loaded")

    print("[load] actor tokenizer", flush=True)
    actor_tokenizer = AutoTokenizer.from_pretrained(args.actor_tokenizer_path, trust_remote_code=True, local_files_only=True)
    print("[load] PRM", flush=True)
    device = resolve_device(args.device)
    prm_tokenizer, prm_model, prm_dtype = load_prm(args.prm_model_path, device)

    piece_cache: dict[int, str] = {}
    examples: list[dict[str, Any]] = []
    sample_infos: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        response = row["response"]
        answer = row["answer"]
        correct = safe_grade(response, answer)
        response_token_ids = actor_tokenizer.encode(response, add_special_tokens=False)
        grouped = split_delimiter_grouped_token_spans(
            response_token_ids,
            actor_tokenizer,
            args.delimiter,
            args.n_prm_blocks,
            piece_cache,
        )
        if not grouped:
            fallback_text = response if response.strip() else " "
            grouped = [(0, len(response_token_ids), fallback_text)]
        block_texts = [block_text for _, _, block_text in grouped]
        prm_input_ids, reward_flags = build_prm_input(prm_tokenizer, row["prompt"], block_texts, args.prm_step_token)
        examples.append({"input_ids": prm_input_ids, "reward_flags": reward_flags})
        sample_infos.append({"num_blocks": len(block_texts), "block_token_spans": [(start, end) for start, end, _ in grouped]})
        records.append(
            {
                "task": row["task"],
                "source_file": row["source_file"],
                "example_id": row["example_id"],
                "seed": row["seed"],
                "correct": correct,
                "score": 1.0 if correct else 0.0,
                "num_blocks": len(block_texts),
                "response_len_chars": len(response),
                "response_len_actor_tokens": len(response_token_ids),
                "prm_input_len": len(prm_input_ids),
                "num_reward_flags": len(reward_flags),
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"[prep] {idx + 1}/{len(rows)}", flush=True)

    print(f"[prm] scoring examples={len(examples)} dtype={prm_dtype} device={device}", flush=True)
    scores_per_example = score_prm_examples(
        examples,
        prm_tokenizer,
        prm_model,
        device,
        prm_dtype,
        args.micro_batch_size,
    )

    running_stats = RunningStats(momentum=0.95)
    all_scores: list[float] = []
    for record, scores in zip(records, scores_per_example):
        if not scores:
            scores = [0.0]
        record["prm_scores"] = scores
        record["num_scores"] = len(scores)
        record.update(compute_features(scores, args))
        add_v2_features(record, scores, running_stats.current(), args, prefix="gate_v2")
        running_stats.update(scores)
        all_scores.extend(scores)

    oracle_mean = mean(all_scores) if all_scores else None
    oracle_std = std(all_scores) if all_scores else None
    oracle_count = len(all_scores)
    for record in records:
        add_v2_features(record, record["prm_scores"], (oracle_mean, oracle_std, oracle_count), args, prefix="gate_v2_oracle")

    summary_rows = compute_summary(records, tasks)
    ece_rows, ece_by_key = compute_ece(records, tasks, SCORE_LIKE_FEATURES, args.ece_bins)
    for row in summary_rows:
        row["ece"] = ece_by_key.get((row["group"], row["feature"]), float("nan"))
    block_curve_rows = compute_block_curves(records, tasks, args.curve_bins)

    write_jsonl(out_dir / "rollout_features.jsonl", records)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "calibration_bins.csv", ece_rows)
    write_csv(out_dir / "block_curves.csv", block_curve_rows)
    write_summary_md(out_dir / "summary.md", records, summary_rows, tasks)

    print(f"[done] wrote report to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
