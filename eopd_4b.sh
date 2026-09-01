#!/usr/bin/env bash

# EOPD reproduction for 2603.07079v3. All non-method parameters inherit
# opd_4b_qwen3_teacher_student.sh.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export METHOD=${METHOD:-eopd}
export EOPD_ENABLE=${EOPD_ENABLE:-True}
export EOPD_ENTROPY_THRESHOLD=${EOPD_ENTROPY_THRESHOLD:-0.8}
export EOPD_FKL_COEF=${EOPD_FKL_COEF:-1.0}
export G_OPD_ENABLE=False
export PRUNE_OPD_ENABLE=False

# This 31 GiB host has a dedicated 64 GiB swapfile. Let the kernel spill cold
# pages instead of allowing Ray's physical-memory monitor to kill EOPD workers.
export RAY_memory_monitor_refresh_ms=0

# EOPD evaluation uses a longer generation budget and more samples per prompt.
# Training rollout settings remain inherited from the baseline (7168 tokens, n=4).
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-32768}
export VAL_N=${VAL_N:-16}

exec bash "$SCRIPT_DIR/opd_4b_qwen3_teacher_student.sh" "$@"
