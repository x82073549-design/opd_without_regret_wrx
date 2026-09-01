#!/usr/bin/env bash

# ExOPD reproduction for 2602.12125v1. All non-method parameters inherit
# opd_4b_qwen3_teacher_student.sh.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export METHOD=${METHOD:-exopd}
export EOPD_ENABLE=False
export G_OPD_ENABLE=${G_OPD_ENABLE:-True}
export G_OPD_REWARD_SCALE=${G_OPD_REWARD_SCALE:-1.25}
export PRUNE_OPD_ENABLE=False

# Storage defaults are inherited from opd_4b_qwen3_teacher_student.sh:
# lightweight text logs use a dedicated repository directory, while
# checkpoints, Ray state, validation generations, and service caches are
# written under /data/opd_outputs_qwen3_4b_teacher_student.

exec bash "$SCRIPT_DIR/opd_4b_qwen3_teacher_student.sh" "$@"
