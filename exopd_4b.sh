#!/usr/bin/env bash

# ExOPD reproduction for 2602.12125v1. All non-method parameters inherit opd_4b.sh.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export METHOD=${METHOD:-exopd}
export G_OPD_ENABLE=${G_OPD_ENABLE:-True}
export G_OPD_REWARD_SCALE=${G_OPD_REWARD_SCALE:-1.25}
export PRUNE_OPD_ENABLE=False

# Keep large run artifacts on /data1; retain human-readable logs in the repo.
export OUTPUT_ROOT=${OUTPUT_ROOT:-/data1/ruxin/OPD-without-regret/outputs/qwen3_4b_rl_math_opd_100step}
export LOG_ROOT=${LOG_ROOT:-$SCRIPT_DIR/outputs/qwen3_4b_rl_math_opd_100step/logs/internal}
export RAY_TMP_DIR=${RAY_TMP_DIR:-/data1/ruxin/ray/gopd}

exec bash "$SCRIPT_DIR/opd_4b.sh" "$@"
