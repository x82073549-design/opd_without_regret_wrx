# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import apply_token_feature_weighting, compute_token_feature_weights


def test_uniform_weights_are_one():
    feature_values = torch.tensor([[0.9, 0.8, 0.1], [0.2, 0.5, 0.7]], dtype=torch.float32)
    response_mask = torch.ones_like(feature_values)

    weights, diagnostics = compute_token_feature_weights(
        feature_values=feature_values,
        response_mask=response_mask,
        alpha=2.0,
        direction="uniform",
    )

    assert torch.allclose(weights, torch.ones_like(weights))
    assert diagnostics["weight_mean"].item() == pytest.approx(1.0)


def test_normal_and_reverse_are_center_symmetric_without_clipping():
    feature_values = torch.tensor([[0.9, 0.8, 0.7, 0.1]], dtype=torch.float32)
    response_mask = torch.ones_like(feature_values)

    normal_weights, _ = compute_token_feature_weights(
        feature_values=feature_values,
        response_mask=response_mask,
        alpha=1.0,
        direction="normal",
        min_weight=-10.0,
    )
    reverse_weights, _ = compute_token_feature_weights(
        feature_values=feature_values,
        response_mask=response_mask,
        alpha=1.0,
        direction="reverse",
        min_weight=-10.0,
    )

    assert torch.allclose(normal_weights + reverse_weights, torch.full_like(normal_weights, 2.0), atol=1e-6)
    assert torch.allclose(normal_weights.mean(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(reverse_weights.mean(dim=-1), torch.ones(1), atol=1e-6)


def test_invalid_tokens_do_not_affect_sequence_statistics():
    feature_values = torch.tensor([[0.9, 0.1, 100.0]], dtype=torch.float32)
    response_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

    weights, _ = compute_token_feature_weights(
        feature_values=feature_values,
        response_mask=response_mask,
        alpha=1.0,
        direction="normal",
        min_weight=-10.0,
    )
    expected_valid, _ = compute_token_feature_weights(
        feature_values=feature_values[:, :2],
        response_mask=response_mask[:, :2],
        alpha=1.0,
        direction="normal",
        min_weight=-10.0,
    )

    assert torch.allclose(weights[:, :2], expected_valid, atol=1e-6)
    assert weights[0, 2].item() == pytest.approx(1.0)
    assert torch.allclose((weights * response_mask).sum(dim=-1) / response_mask.sum(dim=-1), torch.ones(1))


def test_apply_teacher_confidence_weighting_broadcasts_to_topk_rewards():
    teacher_probs = torch.tensor([[[0.8, 0.1], [0.5, 0.2], [0.1, 0.05]]], dtype=torch.float32)
    rewards = torch.tensor([[[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]]], dtype=torch.float32)
    response_mask = torch.ones(1, 3, dtype=torch.float32)
    data = DataProto.from_dict(
        tensors={
            "teacher_top_k_log_probs": torch.log(teacher_probs),
            "response_mask": response_mask,
            "token_level_rewards": rewards.clone(),
        }
    )
    config = {
        "enable": True,
        "feature": "teacher_confidence",
        "alpha": 1.0,
        "direction": "normal",
        "min_weight": 0.0,
    }

    data, metrics = apply_token_feature_weighting(data, config)
    expected_weights, _ = compute_token_feature_weights(
        feature_values=teacher_probs[..., 0],
        response_mask=response_mask,
        alpha=1.0,
        direction="normal",
        min_weight=0.0,
    )

    assert torch.allclose(data.batch["token_level_rewards"], rewards * expected_weights.unsqueeze(-1), atol=1e-6)
    assert "token_feature/teacher_confidence_mean" in metrics
    assert metrics["token_feature/weight_mean"] == pytest.approx(1.0)


def test_teacher_confidence_requires_teacher_topk_log_probs():
    data = DataProto.from_dict(
        tensors={
            "response_mask": torch.ones(1, 2),
            "token_level_rewards": torch.ones(1, 2),
        }
    )

    with pytest.raises(ValueError, match="teacher_top_k_log_probs"):
        apply_token_feature_weighting(
            data,
            {
                "enable": True,
                "feature": "teacher_confidence",
                "alpha": 1.0,
                "direction": "normal",
            },
        )
