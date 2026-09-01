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

import pytest
import torch

from verl.trainer.ppo.opd_variants import (
    PruneOPDLengthController,
    apply_g_opd_reward,
    compute_eopd_fkl_loss,
)


def test_eopd_fkl_renormalizes_teacher_topk_and_gates_by_entropy() -> None:
    student = torch.log(torch.tensor([[[0.5, 0.25], [0.4, 0.4]]], dtype=torch.float32))
    teacher = torch.log(torch.tensor([[[0.6, 0.3], [0.5, 0.3]]], dtype=torch.float32))
    teacher_entropy = torch.tensor([[0.8, 0.2]])
    response_mask = torch.ones((1, 2))

    loss, metrics = compute_eopd_fkl_loss(
        student,
        teacher,
        teacher_entropy,
        response_mask,
        entropy_threshold=0.8,
        fkl_coef=1.0,
    )

    teacher_normalized = torch.tensor([2.0 / 3.0, 1.0 / 3.0])
    expected_high_token_fkl = torch.sum(
        teacher_normalized * (torch.log(teacher_normalized) - student[0, 0])
    )
    torch.testing.assert_close(loss, expected_high_token_fkl / 2.0)
    torch.testing.assert_close(metrics["high_entropy_ratio"], torch.tensor(0.5))
    torch.testing.assert_close(metrics["high_entropy_fkl"], expected_high_token_fkl)
    torch.testing.assert_close(metrics["teacher_topk_mass"], torch.tensor(0.85))


def test_eopd_fkl_gradient_matches_teacher_to_student_kl() -> None:
    logits = torch.tensor([[[0.4, -0.2, 0.1]]], requires_grad=True)
    student_log_probs = torch.log_softmax(logits, dim=-1)
    teacher_probs = torch.tensor([[[0.2, 0.3, 0.5]]])

    loss, _ = compute_eopd_fkl_loss(
        student_log_probs,
        torch.log(teacher_probs),
        teacher_entropy=torch.tensor([[1.0]]),
        response_mask=torch.ones((1, 1)),
    )
    loss.backward()

    expected_gradient = torch.softmax(logits.detach(), dim=-1) - teacher_probs
    torch.testing.assert_close(logits.grad, expected_gradient)


def test_eopd_fkl_low_entropy_has_zero_loss_and_gradient() -> None:
    logits = torch.tensor([[[0.4, -0.2]]], requires_grad=True)
    student_log_probs = torch.log_softmax(logits, dim=-1)

    loss, metrics = compute_eopd_fkl_loss(
        student_log_probs,
        torch.log(torch.tensor([[[0.6, 0.4]]])),
        teacher_entropy=torch.tensor([[0.79]]),
        response_mask=torch.ones((1, 1)),
        entropy_threshold=0.8,
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
    torch.testing.assert_close(metrics["high_entropy_ratio"], torch.tensor(0.0))


def test_eopd_fkl_ignores_padded_high_entropy_tokens() -> None:
    student = torch.log(torch.tensor([[[0.5, 0.25], [0.01, 0.01]]]))
    teacher = torch.log(torch.tensor([[[0.6, 0.4], [0.9, 0.1]]]))

    loss, metrics = compute_eopd_fkl_loss(
        student,
        teacher,
        teacher_entropy=torch.tensor([[1.0, 10.0]]),
        response_mask=torch.tensor([[1.0, 0.0]]),
    )

    expected = torch.sum(torch.tensor([0.6, 0.4]) * (torch.log(torch.tensor([0.6, 0.4])) - student[0, 0]))
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["high_entropy_ratio"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["teacher_topk_mass"], torch.tensor(1.0))


def test_eopd_fkl_rejects_invalid_configuration() -> None:
    student = torch.zeros((1, 2, 3))
    teacher = torch.zeros_like(student)
    entropy = torch.zeros((1, 2))
    mask = torch.ones((1, 2))

    with pytest.raises(ValueError, match="entropy_threshold"):
        compute_eopd_fkl_loss(student, teacher, entropy, mask, entropy_threshold=-0.1)
    with pytest.raises(ValueError, match="identical shapes"):
        compute_eopd_fkl_loss(student, teacher[..., :2], entropy, mask)


def test_g_opd_lambda_one_is_baseline() -> None:
    student = torch.log(torch.tensor([[[0.7, 0.3]]]))
    teacher = torch.log(torch.tensor([[[0.4, 0.6]]]))
    reference = torch.log(torch.tensor([[[0.8, 0.2]]]))
    baseline = torch.tensor([[[1.0, -2.0]]])

    corrected, correction = apply_g_opd_reward(baseline, student, teacher, reference, reward_scale=1.0)

    torch.testing.assert_close(corrected, baseline)
    torch.testing.assert_close(correction, torch.zeros_like(correction))


def test_g_opd_matches_extrapolated_target_distribution() -> None:
    student = torch.log(torch.tensor([[[0.7, 0.3]]]))
    teacher = torch.log(torch.tensor([[[0.4, 0.6]]]))
    reference = torch.log(torch.tensor([[[0.8, 0.2]]]))
    weights = torch.softmax(student, dim=-1)
    baseline = (teacher - student) * weights

    corrected, _ = apply_g_opd_reward(baseline, student, teacher, reference, reward_scale=1.25)
    expected = (1.25 * teacher - 0.25 * reference - student) * weights

    torch.testing.assert_close(corrected, expected)


def test_prune_opd_length_controller_expands_and_shrinks() -> None:
    controller = PruneOPDLengthController(
        initial_length=4,
        min_length=2,
        max_length=6,
        step=2,
        margin=0,
        hit_ratio_threshold=0.5,
        shrink_patience=2,
    )
    response_mask = torch.ones((2, 4))

    expanded = controller.update(torch.ones((2, 4)), response_mask)
    assert expanded["used_length"] == 4
    assert expanded["next_length"] == 6

    low_raw_weight = torch.zeros((2, 4))
    first_low = controller.update(low_raw_weight, response_mask)
    second_low = controller.update(low_raw_weight, response_mask)
    assert first_low["next_length"] == 6
    assert second_low["next_length"] == 4


def test_prune_opd_length_controller_hit_boundary_respects_min_length() -> None:
    controller = PruneOPDLengthController(
        initial_length=4,
        min_length=4,
        max_length=6,
        step=1,
        margin=1,
        hit_ratio_threshold=0.5,
    )
    response_mask = torch.ones((1, 4))
    raw_weight = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

    result = controller.update(raw_weight, response_mask)

    assert result["hit_ratio"] == 0.0
    assert result["next_length"] == 4


def test_prune_opd_length_controller_restores_state() -> None:
    controller = PruneOPDLengthController(initial_length=4, min_length=2, max_length=6)
    controller.load_state_dict({"current_length": 5, "low_hit_steps": 1})
    assert controller.state_dict() == {"current_length": 5, "low_hit_steps": 1}
