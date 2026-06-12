"""Oracle recoverability-aware OPD helper.

This module builds teacher-continuation requests from student prefixes, sends
those requests to a Ray-managed vLLM teacher actor, grades the continuations,
and returns a token-level gate that can be multiplied into OPD rewards.
"""

from __future__ import annotations

import gc
import math
import time
from collections import defaultdict
from typing import Any

import ray
import torch

from verl.utils.reward_score.ttrl_math import compute_score


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class _OracleRATeacherActor:
    """Ray-managed vLLM teacher actor for recoverability continuations."""

    def __init__(self, config: dict[str, Any]):
        from vllm import LLM, SamplingParams

        self.config = dict(config)
        self._sampling_params_cls = SamplingParams

        model_path = self.config.get("teacher_model_path")
        if not model_path:
            raise ValueError("algorithm.oracle_ra_opd.teacher_model_path must be set")

        actor_rank = int(self.config.get("actor_rank", 0))
        tensor_parallel_size = int(self.config.get("tensor_parallel_size", 1))
        teacher_gpus_per_actor = int(self.config.get("teacher_gpus_per_actor", tensor_parallel_size))
        if teacher_gpus_per_actor < tensor_parallel_size:
            raise ValueError(
                "algorithm.oracle_ra_opd.teacher_gpus_per_actor must be >= tensor_parallel_size, "
                f"got teacher_gpus_per_actor={teacher_gpus_per_actor}, tensor_parallel_size={tensor_parallel_size}"
            )

        llm_kwargs = {
            "model": model_path,
            "trust_remote_code": True,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": float(self.config.get("gpu_memory_utilization", 0.8)),
            "enforce_eager": _as_bool(self.config.get("enforce_eager", True)),
        }
        max_model_len = self.config.get("max_model_len", None)
        if max_model_len not in (None, "", "null"):
            llm_kwargs["max_model_len"] = int(max_model_len)

        print(
            "[oracle_ra] Ray teacher actor loading vLLM "
            f"rank={actor_rank} model={model_path} tp={tensor_parallel_size} "
            f"gpus_per_actor={teacher_gpus_per_actor} "
            f"gpu_memory_utilization={llm_kwargs['gpu_memory_utilization']} "
            f"enforce_eager={llm_kwargs['enforce_eager']} max_model_len={llm_kwargs.get('max_model_len')}",
            flush=True,
        )
        self._llm = LLM(**llm_kwargs)
        self._llm_tokenizer = self._llm.get_tokenizer()
        self._stop_token_ids = []
        for stop_token in ["<|im_end|>", "<|endoftext|>"]:
            try:
                encoded = self._llm_tokenizer.encode(stop_token, add_special_tokens=False)
                if encoded:
                    self._stop_token_ids.append(int(encoded[0]))
            except Exception:
                pass
        print(f"[oracle_ra] Ray teacher actor rank={actor_rank} vLLM loaded", flush=True)

    @staticmethod
    def _score_correct(response: str, answer: Any) -> bool:
        result = compute_score(response, answer)
        if isinstance(result, dict):
            if "acc" in result:
                return bool(result["acc"])
            return float(result.get("score", 0.0)) > 0.5
        return bool(result)

    def run(self, requests: list[dict[str, Any]]) -> dict[str, float]:
        if not requests:
            return {}

        num_continuations = int(self.config.get("num_continuations", 16))
        temperature = float(self.config.get("temperature", 0.7))
        top_p = float(self.config.get("top_p", 0.95))
        generation_batch_size = int(self.config.get("generation_batch_size", 1))
        if generation_batch_size <= 0:
            generation_batch_size = len(requests)

        success: dict[str, list[float]] = defaultdict(list)
        for start in range(0, len(requests), generation_batch_size):
            chunk = requests[start : start + generation_batch_size]
            prompts = [{"prompt_token_ids": req["prompt_token_ids"]} for req in chunk]
            sampling_params = [
                self._sampling_params_cls(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=int(req["max_new_tokens"]),
                    n=num_continuations,
                    stop_token_ids=self._stop_token_ids if self._stop_token_ids else None,
                )
                for req in chunk
            ]
            outputs = self._llm.generate(prompts, sampling_params, use_tqdm=False)
            for req, out in zip(chunk, outputs, strict=True):
                if len(out.outputs) != num_continuations:
                    raise RuntimeError(
                        f"oracle_ra teacher returned {len(out.outputs)} continuations for {req['request_id']}, "
                        f"expected {num_continuations}"
                    )
                for sample in out.outputs:
                    full_response = req["prefix_text"] + sample.text
                    success[req["request_id"]].append(
                        1.0 if self._score_correct(full_response, req["answer"]) else 0.0
                    )

        return {request_id: sum(values) / len(values) for request_id, values in success.items()}


class OracleRAOPDHelper:
    """Compute online oracle recoverability gates for OPD rewards."""

    def __init__(self, config: dict[str, Any], tokenizer: Any, max_response_length: int):
        self.config = dict(config)
        self.tokenizer = tokenizer
        self.max_response_length = int(max_response_length)
        self._teacher_actors: list[Any] = []
        self._last_dispatch_metrics: dict[str, float] = {}

    @staticmethod
    def enabled(config: dict[str, Any] | None) -> bool:
        return bool(config and config.get("enable", False))

    @staticmethod
    def _parse_prefix_fracs(value: Any) -> list[float]:
        if isinstance(value, str):
            fracs = [float(item.strip()) for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple)):
            fracs = [float(item) for item in value]
        else:
            raise ValueError(f"Unsupported oracle_ra_opd.prefix_fracs={value!r}")
        if not fracs:
            raise ValueError("algorithm.oracle_ra_opd.prefix_fracs must not be empty")
        for frac in fracs:
            if frac < 0.0 or frac >= 1.0:
                raise ValueError(f"oracle_ra_opd prefix fraction must be in [0, 1), got {frac}")
        return sorted(set(fracs))

    @staticmethod
    def _frac_label(frac: float) -> str:
        pct = frac * 100.0
        if abs(pct - round(pct)) < 1e-6:
            return str(int(round(pct)))
        return f"{pct:g}".replace(".", "p")

    @staticmethod
    def _extract_ground_truth(item: Any) -> Any | None:
        if not isinstance(item, dict):
            return None
        answer = item.get("ground_truth", None)
        if hasattr(answer, "tolist"):
            answer = answer.tolist()
        if hasattr(answer, "item") and not isinstance(answer, (str, bytes, list, tuple, dict)):
            try:
                answer = answer.item()
            except Exception:
                pass
        return answer

    def _prompt_token_ids(self, prompt_ids: torch.Tensor) -> list[int]:
        prompt_ids = prompt_ids.detach().cpu()
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is not None:
            non_pad = torch.nonzero(prompt_ids != pad_token_id, as_tuple=False)
            if non_pad.numel() == 0:
                return []
            prompt_ids = prompt_ids[int(non_pad[0].item()) :]
        return [int(token_id) for token_id in prompt_ids.tolist()]

    def _teacher_pool_config(self) -> tuple[int, int, int, str]:
        num_actors = int(self.config.get("teacher_num_actors", 1))
        gpus_per_actor = int(self.config.get("teacher_gpus_per_actor", 1))
        tensor_parallel_size = int(self.config.get("tensor_parallel_size", 1))
        dispatch_policy = str(self.config.get("dispatch_policy", "length_bucket_round_robin"))

        if num_actors <= 0:
            raise ValueError(f"oracle_ra_opd.teacher_num_actors must be positive, got {num_actors}")
        if gpus_per_actor <= 0:
            raise ValueError(f"oracle_ra_opd.teacher_gpus_per_actor must be positive, got {gpus_per_actor}")
        if tensor_parallel_size <= 0:
            raise ValueError(f"oracle_ra_opd.tensor_parallel_size must be positive, got {tensor_parallel_size}")
        if tensor_parallel_size > gpus_per_actor:
            raise ValueError(
                "oracle_ra_opd.tensor_parallel_size must be <= teacher_gpus_per_actor, "
                f"got tensor_parallel_size={tensor_parallel_size}, teacher_gpus_per_actor={gpus_per_actor}"
            )
        if dispatch_policy not in {"length_bucket_round_robin", "round_robin"}:
            raise ValueError(f"Unsupported oracle_ra_opd.dispatch_policy={dispatch_policy!r}")

        return num_actors, gpus_per_actor, tensor_parallel_size, dispatch_policy

    def _ensure_teacher_actors(self):
        if self._teacher_actors:
            return

        num_actors, gpus_per_actor, tensor_parallel_size, dispatch_policy = self._teacher_pool_config()
        print(
            "[oracle_ra] creating Ray teacher actor pool "
            f"num_actors={num_actors} gpus_per_actor={gpus_per_actor} "
            f"tensor_parallel_size={tensor_parallel_size} dispatch_policy={dispatch_policy}",
            flush=True,
        )
        actor_cls = ray.remote(num_gpus=gpus_per_actor)(_OracleRATeacherActor)
        for actor_rank in range(num_actors):
            actor_config = dict(self.config)
            actor_config["actor_rank"] = actor_rank
            actor_config["teacher_gpus_per_actor"] = gpus_per_actor
            actor_config["tensor_parallel_size"] = tensor_parallel_size
            self._teacher_actors.append(actor_cls.remote(actor_config))

    def close(self):
        for actor in self._teacher_actors:
            ray.kill(actor, no_restart=True)
        self._teacher_actors = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cap_max_new_tokens(self, max_new_tokens: int) -> int:
        max_new_tokens = max(1, int(max_new_tokens))
        cap = self.config.get("max_new_tokens_cap", None)
        if cap in (None, "", "null"):
            return max_new_tokens
        cap = int(cap)
        if cap <= 0:
            raise ValueError(f"oracle_ra_opd.max_new_tokens_cap must be positive or null, got {cap}")
        return max(1, min(max_new_tokens, cap))

    def _build_requests(
        self,
        batch: Any,
        prefix_fracs: list[float],
        max_new_tokens_base: int,
    ) -> tuple[list[dict[str, Any]], list[Any | None], int]:
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        prompts = batch.batch["prompts"]
        reward_model_info = batch.non_tensor_batch.get("reward_model", None)
        batch_size, _ = responses.shape

        ground_truths: list[Any | None] = []
        skipped_no_answer = 0
        if reward_model_info is None:
            ground_truths = [None] * batch_size
            skipped_no_answer = batch_size
        else:
            for row in range(batch_size):
                answer = self._extract_ground_truth(reward_model_info[row])
                ground_truths.append(answer)
                if answer is None:
                    skipped_no_answer += 1

        positive_fracs = [frac for frac in prefix_fracs if frac > 0.0]
        requests: list[dict[str, Any]] = []
        for row in range(batch_size):
            answer = ground_truths[row]
            if answer is None:
                continue
            valid_len = int(response_mask[row].sum().item())
            prompt_token_ids = self._prompt_token_ids(prompts[row])
            response_token_ids = [int(token_id) for token_id in responses[row, :valid_len].detach().cpu().tolist()]

            requests.append(
                {
                    "request_id": f"{row}:clean",
                    "row": row,
                    "frac": 0.0,
                    "answer": answer,
                    "prefix_text": "",
                    "prompt_token_ids": prompt_token_ids,
                    "max_new_tokens": self._cap_max_new_tokens(int(max_new_tokens_base)),
                }
            )

            if valid_len <= 0:
                continue

            for frac in positive_fracs:
                prefix_len = int(math.floor(valid_len * frac))
                prefix_len = max(1, min(valid_len, prefix_len))
                prefix_ids = response_token_ids[:prefix_len]
                prefix_text = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)
                requests.append(
                    {
                        "request_id": f"{row}:f{self._frac_label(frac)}",
                        "row": row,
                        "frac": frac,
                        "answer": answer,
                        "prefix_text": prefix_text,
                        "prompt_token_ids": prompt_token_ids + prefix_ids,
                        "max_new_tokens": self._cap_max_new_tokens(int(max_new_tokens_base) - prefix_len),
                    }
                )

        return requests, ground_truths, skipped_no_answer

    def _dispatch_requests(self, requests: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], list[int]]:
        num_actors, _, _, dispatch_policy = self._teacher_pool_config()
        shards: list[list[dict[str, Any]]] = [[] for _ in range(num_actors)]
        loads = [0 for _ in range(num_actors)]
        num_continuations = max(1, int(self.config.get("num_continuations", 16)))

        if dispatch_policy == "round_robin":
            for idx, req in enumerate(requests):
                actor_idx = idx % num_actors
                load = max(1, int(req.get("max_new_tokens", 1))) * num_continuations
                shards[actor_idx].append(req)
                loads[actor_idx] += load
            return shards, loads

        sorted_requests = sorted(
            requests,
            key=lambda req: max(1, int(req.get("max_new_tokens", 1))) * num_continuations,
            reverse=True,
        )
        for req in sorted_requests:
            actor_idx = min(range(num_actors), key=lambda idx: loads[idx])
            load = max(1, int(req.get("max_new_tokens", 1))) * num_continuations
            shards[actor_idx].append(req)
            loads[actor_idx] += load
        return shards, loads

    def _run_teacher(self, requests: list[dict[str, Any]]) -> dict[str, float]:
        num_actors, gpus_per_actor, tensor_parallel_size, _ = self._teacher_pool_config()
        self._last_dispatch_metrics = {
            "oracle_ra/teacher_actor_count": float(num_actors),
            "oracle_ra/teacher_gpus_per_actor": float(gpus_per_actor),
            "oracle_ra/teacher_tensor_parallel_size": float(tensor_parallel_size),
            "oracle_ra/request_count": float(len(requests)),
            "oracle_ra/teacher_active_actor_count": 0.0,
            "oracle_ra/teacher_max_shard_load": 0.0,
            "oracle_ra/teacher_min_shard_load": 0.0,
        }
        if not requests:
            return {}
        self._ensure_teacher_actors()
        shards, loads = self._dispatch_requests(requests)
        futures = [
            actor.run.remote(shard)
            for actor, shard in zip(self._teacher_actors, shards, strict=True)
            if shard
        ]
        results: dict[str, float] = {}
        for shard_result in ray.get(futures):
            overlap = set(results).intersection(shard_result)
            if overlap:
                raise RuntimeError(f"oracle_ra duplicate teacher request ids: {sorted(overlap)[:3]}")
            results.update(shard_result)

        self._last_dispatch_metrics.update(
            {
                "oracle_ra/teacher_active_actor_count": float(sum(1 for shard in shards if shard)),
                "oracle_ra/teacher_max_shard_load": float(max(loads) if loads else 0),
                "oracle_ra/teacher_min_shard_load": float(min(loads) if loads else 0),
            }
        )
        return results

    def compute_gate(self, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        start_time = time.time()
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        batch_size, response_length = responses.shape

        prefix_fracs = self._parse_prefix_fracs(self.config.get("prefix_fracs", "0.0,0.1,0.25,0.5,0.75"))
        positive_fracs = [frac for frac in prefix_fracs if frac > 0.0]
        max_tokens_mode = self.config.get("max_tokens_mode", "train_response_length")
        if max_tokens_mode != "train_response_length":
            raise ValueError(f"Unsupported oracle_ra_opd.max_tokens_mode={max_tokens_mode}")
        max_new_tokens_base = self.max_response_length

        gate = torch.ones((batch_size, response_length), dtype=torch.float32)
        requests, ground_truths, skipped_no_answer = self._build_requests(batch, prefix_fracs, max_new_tokens_base)
        rho_by_request = self._run_teacher(requests)

        epsilon = float(self.config.get("epsilon", 1e-6))
        gate_score_mode = str(self.config.get("gate_score_mode", "relative"))
        min_gate = float(self.config.get("min_gate", 0.0))
        gate_transform = str(self.config.get("gate_transform", "clip"))
        if gate_score_mode not in {"relative", "absolute"}:
            raise ValueError(f"Unsupported oracle_ra_opd.gate_score_mode={gate_score_mode!r}")
        if gate_transform not in {"clip", "affine_floor"}:
            raise ValueError(f"Unsupported oracle_ra_opd.gate_transform={gate_transform!r}")
        if min_gate < 0.0 or min_gate > 1.0:
            raise ValueError(f"oracle_ra_opd.min_gate must be in [0, 1], got {min_gate}")
        rho0_values = []
        rho_by_frac: dict[float, list[float]] = {frac: [] for frac in positive_fracs}
        gate_values = []
        rho0_zero = 0

        for row in range(batch_size):
            valid_len = int(response_mask[row].sum().item())
            if ground_truths[row] is None or valid_len <= 0:
                continue

            rho0 = float(rho_by_request.get(f"{row}:clean", 0.0))
            rho0_values.append(rho0)
            if rho0 <= 0.0:
                rho0_zero += 1

            row_gate_by_frac = {}
            for frac in positive_fracs:
                rho = float(rho_by_request.get(f"{row}:f{self._frac_label(frac)}", 0.0))
                rho_by_frac[frac].append(rho)
                if gate_score_mode == "relative":
                    raw_gate = rho / (rho0 + epsilon)
                else:
                    raw_gate = rho
                clipped_gate = min(1.0, max(0.0, raw_gate))
                if gate_transform == "affine_floor":
                    gate_value = min_gate + (1.0 - min_gate) * clipped_gate
                else:
                    gate_value = max(min_gate, clipped_gate)
                row_gate_by_frac[frac] = gate_value

            if not positive_fracs:
                continue

            boundaries = {frac: int(math.floor(valid_len * frac)) for frac in positive_fracs}
            sorted_fracs = sorted(positive_fracs)
            first_boundary = max(0, min(valid_len, boundaries[sorted_fracs[0]]))
            gate[row, :first_boundary] = 1.0
            for idx, frac in enumerate(sorted_fracs):
                start = max(0, min(valid_len, boundaries[frac]))
                if idx + 1 < len(sorted_fracs):
                    end = max(start, min(valid_len, boundaries[sorted_fracs[idx + 1]]))
                else:
                    end = valid_len
                gate_value = float(row_gate_by_frac.get(frac, 1.0))
                gate[row, start:end] = gate_value
                if end > start:
                    gate_values.append(gate_value)

        response_mask_float = response_mask.float().cpu()
        mask_denom = response_mask_float.sum().clamp_min(1.0)
        valid_gate = gate * response_mask_float
        gate_mean = (valid_gate.sum() / mask_denom).item()
        gate_var = (((gate - gate_mean) ** 2) * response_mask_float).sum() / mask_denom
        valid_values = gate[response_mask_float.bool()]

        metrics: dict[str, float] = {
            "oracle_ra/rho0_mean": sum(rho0_values) / len(rho0_values) if rho0_values else 0.0,
            "oracle_ra/rho0_zero_ratio": rho0_zero / len(rho0_values) if rho0_values else 0.0,
            "oracle_ra/gate_mean": gate_mean,
            "oracle_ra/gate_std": torch.sqrt(gate_var.clamp_min(0.0)).item(),
            "oracle_ra/gate_low_ratio_0.5": (((gate < 0.5).float() * response_mask_float).sum() / mask_denom).item(),
            "oracle_ra/gate_min": valid_values.min().item() if valid_values.numel() > 0 else 1.0,
            "oracle_ra/gate_max": valid_values.max().item() if valid_values.numel() > 0 else 1.0,
            "oracle_ra/teacher_completion_count": float(len(requests) * int(self.config.get("num_continuations", 16))),
            "oracle_ra/latency_sec": time.time() - start_time,
            "oracle_ra/skipped_no_answer": float(skipped_no_answer),
            "oracle_ra/min_gate": min_gate,
            "oracle_ra/gate_score_mode_absolute": 1.0 if gate_score_mode == "absolute" else 0.0,
            "oracle_ra/gate_transform_affine_floor": 1.0 if gate_transform == "affine_floor" else 0.0,
        }
        max_new_tokens_cap = self.config.get("max_new_tokens_cap", None)
        if max_new_tokens_cap not in (None, "", "null"):
            metrics["oracle_ra/max_new_tokens_cap"] = float(max_new_tokens_cap)
        metrics.update(self._last_dispatch_metrics)
        for frac, values in rho_by_frac.items():
            label = self._frac_label(frac)
            metrics[f"oracle_ra/rho_{label}_mean"] = sum(values) / len(values) if values else 0.0

        return gate.to(device=responses.device), metrics
