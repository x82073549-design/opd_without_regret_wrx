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

import torch

from verl.trainer.ppo.opd_variants import PruneOPDLengthController, apply_g_opd_reward


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
