# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import torch


def apply_g_opd_reward(
    base_rewards: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    reward_scale: float,
    reward_weight_mode: str = "student_p",
    candidate_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the G-OPD/ExOPD correction on student top-k candidates.

    The baseline top-k OPD reward is ``p_student * (log p_teacher - log p_student)``.
    Equation (14) of the G-OPD paper adds
    ``(lambda - 1) * p_student * (log p_teacher - log p_reference)``.
    """
    tensors = (base_rewards, student_log_probs, teacher_log_probs, reference_log_probs)
    if any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("G-OPD expects [batch, response_length, top_k] tensors")
    if any(tensor.shape != base_rewards.shape for tensor in tensors[1:]):
        raise ValueError("G-OPD reward and log-probability tensors must have identical shapes")
    if reward_scale <= 0:
        raise ValueError(f"G-OPD reward_scale must be positive, got {reward_scale}")

    if candidate_mask is None:
        candidate_mask = torch.ones_like(student_log_probs, dtype=torch.bool)
    else:
        candidate_mask = candidate_mask.to(device=student_log_probs.device, dtype=torch.bool)
        if candidate_mask.shape != student_log_probs.shape:
            raise ValueError("G-OPD candidate_mask must match the top-k log-probability shape")

    if reward_weight_mode == "student_p":
        weight_log_probs = student_log_probs
    elif reward_weight_mode == "teacher_p":
        weight_log_probs = teacher_log_probs
    elif reward_weight_mode == "none":
        weight_log_probs = torch.zeros_like(student_log_probs)
    else:
        raise ValueError(f"Unsupported G-OPD reward_weight_mode: {reward_weight_mode}")

    masked_log_probs = torch.where(
        candidate_mask,
        weight_log_probs,
        torch.full_like(weight_log_probs, -float("inf")),
    )
    normalized_log_weights = masked_log_probs - torch.logsumexp(masked_log_probs, dim=-1, keepdim=True)
    weights = torch.where(candidate_mask, torch.exp(normalized_log_weights), torch.zeros_like(masked_log_probs))
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    correction = (reward_scale - 1.0) * (teacher_log_probs - reference_log_probs) * weights
    correction = torch.where(candidate_mask, correction, torch.zeros_like(correction))
    correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    corrected_rewards = torch.nan_to_num(
        base_rewards + correction.to(device=base_rewards.device, dtype=base_rewards.dtype),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return corrected_rewards, correction


@dataclass
class PruneOPDLengthController:
    """Paper-faithful dynamic response-budget controller for Prune-OPD."""

    initial_length: int
    min_length: int
    max_length: int
    step: int = 100
    margin: int = 100
    hit_ratio_threshold: float = 0.1
    shrink_patience: int = 3
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not 0 < self.min_length <= self.initial_length <= self.max_length:
            raise ValueError("Prune-OPD lengths must satisfy 0 < min <= initial <= max")
        if self.step <= 0:
            raise ValueError("Prune-OPD length step must be positive")
        if self.margin < 0:
            raise ValueError("Prune-OPD length margin must be non-negative")
        if not 0 <= self.hit_ratio_threshold <= 1:
            raise ValueError("Prune-OPD hit_ratio_threshold must be in [0, 1]")
        if self.shrink_patience <= 0:
            raise ValueError("Prune-OPD shrink_patience must be positive")
        if self.epsilon < 0:
            raise ValueError("Prune-OPD epsilon must be non-negative")
        self.current_length = self.initial_length
        self.low_hit_steps = 0

    def update(self, raw_weight: torch.Tensor, response_mask: torch.Tensor) -> dict[str, float]:
        if raw_weight.ndim != 2 or response_mask.ndim != 2 or raw_weight.shape != response_mask.shape:
            raise ValueError("Prune-OPD raw_weight and response_mask must be matching [batch, response_length] tensors")

        valid_mask = response_mask.to(device=raw_weight.device, dtype=torch.bool)
        effective_lengths = ((raw_weight > self.epsilon) & valid_mask).sum(dim=-1)
        used_length = self.current_length
        hit_boundary = max(self.min_length, used_length - self.margin)
        hit_ratio = (effective_lengths >= hit_boundary).float().mean().item()

        if hit_ratio >= self.hit_ratio_threshold:
            self.current_length = min(used_length + self.step, self.max_length)
            self.low_hit_steps = 0
        else:
            self.low_hit_steps += 1
            if self.low_hit_steps >= self.shrink_patience:
                self.current_length = max(used_length - self.step, self.min_length)
                self.low_hit_steps = 0

        effective_lengths_float = effective_lengths.float()
        return {
            "used_length": float(used_length),
            "next_length": float(self.current_length),
            "hit_ratio": hit_ratio,
            "effective_length_mean": effective_lengths_float.mean().item(),
            "effective_length_min": effective_lengths_float.min().item(),
            "effective_length_max": effective_lengths_float.max().item(),
            "low_hit_steps": float(self.low_hit_steps),
        }

    def state_dict(self) -> dict[str, int]:
        return {"current_length": self.current_length, "low_hit_steps": self.low_hit_steps}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        current_length = int(state_dict["current_length"])
        low_hit_steps = int(state_dict["low_hit_steps"])
        if not self.min_length <= current_length <= self.max_length:
            raise ValueError("Saved Prune-OPD current_length is outside configured bounds")
        if not 0 <= low_hit_steps < self.shrink_patience:
            raise ValueError("Saved Prune-OPD low_hit_steps is invalid")
        self.current_length = current_length
        self.low_hit_steps = low_hit_steps
