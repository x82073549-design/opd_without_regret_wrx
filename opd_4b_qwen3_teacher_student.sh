#!/usr/bin/env bash

# Qwen3-4B teacher/student launcher for the existing fixed-OPD configuration.
#
# Student (actor): Qwen/Qwen3-4B with thinking disabled.
# Teacher (reward): Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500.
# All other defaults, environment overrides, modes, and positional arguments
# are inherited unchanged from opd_4b.sh.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ACTOR_MODEL_PATH=/data/models/Qwen3-4B
export REWARD_MODEL_PATH=/data/models/Qwen3-4B-Non-Thinking-RL-Math-Step500
export ENABLE_THINKING=False

# Keep this Qwen3-4B teacher/student experiment family isolated from the
# earlier DeepSeek/JustRL runs while retaining every training hyperparameter.
export OUTPUT_ROOT=${OUTPUT_ROOT:-/data/opd_outputs_qwen3_4b_teacher_student}
export LOG_OUTPUT_ROOT=${LOG_OUTPUT_ROOT:-$SCRIPT_DIR/outputs/qwen3_4b_teacher_student}
export PROJECT_NAME=${PROJECT_NAME:-opd_qwen3_4b_teacher_student}
export RAY_LOG_ROOT=${RAY_LOG_ROOT:-$OUTPUT_ROOT/logs/ray_qwen3_4b_teacher_student}

exec "$SCRIPT_DIR/opd_4b.sh" "$@"
