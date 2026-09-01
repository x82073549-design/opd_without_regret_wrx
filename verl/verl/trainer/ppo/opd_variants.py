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

import math
from dataclasses import dataclass

import torch


def compute_eopd_fkl_loss(
    student_log_probs: torch.Tensor,
    teacher_top_k_log_probs: torch.Tensor,
    teacher_entropy: torch.Tensor,
    response_mask: torch.Tensor,
    entropy_threshold: float = 0.8,
    fkl_coef: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the entropy-gated top-k forward-KL term from EOPD.

    ``teacher_top_k_log_probs`` are log probabilities under the teacher's full
    vocabulary distribution. EOPD renormalizes them within the teacher top-k,
    while ``student_log_probs`` remain probabilities from the student's full
    vocabulary distribution, matching Equation (10) of arXiv:2603.07079v3.
    The gated loss is normalized by all valid response tokens so it can be
    added directly to the baseline token-mean OPD loss.
    """
    if student_log_probs.ndim != 3 or teacher_top_k_log_probs.ndim != 3:
        raise ValueError("EOPD top-k log probabilities must be [batch, response_length, top_k] tensors")
    if student_log_probs.shape != teacher_top_k_log_probs.shape:
        raise ValueError("EOPD student and teacher top-k log-probability tensors must have identical shapes")
    if student_log_probs.shape[-1] <= 0:
        raise ValueError("EOPD requires a positive top-k size")
    expected_token_shape = student_log_probs.shape[:-1]
    if teacher_entropy.ndim != 2 or teacher_entropy.shape != expected_token_shape:
        raise ValueError("EOPD teacher_entropy must match the [batch, response_length] token shape")
    if response_mask.ndim != 2 or response_mask.shape != expected_token_shape:
        raise ValueError("EOPD response_mask must match the [batch, response_length] token shape")
    if not math.isfinite(entropy_threshold) or entropy_threshold < 0:
        raise ValueError(f"EOPD entropy_threshold must be finite and non-negative, got {entropy_threshold}")
    if not math.isfinite(fkl_coef) or fkl_coef < 0:
        raise ValueError(f"EOPD fkl_coef must be finite and non-negative, got {fkl_coef}")

    # Keep the probability arithmetic in fp32 even when the models run in
    # bfloat16. Casting the student log probabilities preserves their gradient.
    student_log_probs_fp32 = student_log_probs.float()
    teacher_top_k_log_probs_fp32 = teacher_top_k_log_probs.detach().float()
    teacher_top_k_normalized_log_probs = teacher_top_k_log_probs_fp32 - torch.logsumexp(
        teacher_top_k_log_probs_fp32, dim=-1, keepdim=True
    )
    teacher_top_k_probs = torch.exp(teacher_top_k_normalized_log_probs)

    fkl_per_token = torch.sum(
        teacher_top_k_probs * (teacher_top_k_normalized_log_probs - student_log_probs_fp32), dim=-1
    )
    valid_mask = response_mask.to(device=fkl_per_token.device, dtype=torch.bool)
    high_entropy_mask = (teacher_entropy.detach().to(fkl_per_token.device).float() >= entropy_threshold) & valid_mask
    valid_mask_fp32 = valid_mask.float()
    high_entropy_mask_fp32 = high_entropy_mask.float()
    valid_token_count = valid_mask_fp32.sum().clamp_min(1.0)
    high_entropy_token_count = high_entropy_mask_fp32.sum().clamp_min(1.0)

    fkl_loss = fkl_coef * (fkl_per_token * high_entropy_mask_fp32).sum() / valid_token_count
    teacher_top_k_mass = torch.exp(teacher_top_k_log_probs_fp32).sum(dim=-1)
    metrics = {
        "high_entropy_ratio": high_entropy_mask_fp32.sum() / valid_token_count,
        "high_entropy_fkl": (fkl_per_token.detach() * high_entropy_mask_fp32).sum()
        / high_entropy_token_count,
        "teacher_topk_mass": (teacher_top_k_mass.detach() * valid_mask_fp32).sum() / valid_token_count,
    }
    return fkl_loss, metrics


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
