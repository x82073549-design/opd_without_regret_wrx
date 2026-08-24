#!/usr/bin/env bash

# ExOPD reproduction for 2602.12125v1. All non-method parameters inherit opd_4b.sh.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export METHOD=${METHOD:-exopd}
export G_OPD_ENABLE=${G_OPD_ENABLE:-True}
export G_OPD_REWARD_SCALE=${G_OPD_REWARD_SCALE:-1.25}
export PRUNE_OPD_ENABLE=False

# Keep all artifacts on the current host's workspace volume.  opd_4b.sh creates
# a short /tmp Ray symlink automatically, so do not retain the old /data1 path.
export OUTPUT_ROOT=${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/qwen3_4b_rl_math_exopd}
export LOG_ROOT=${LOG_ROOT:-$OUTPUT_ROOT/logs/internal}

# /tmp and the workspace volume are nearly full on this host.  Put the Ray
# session and spill files on the independent shared-memory filesystem instead.
RAY_RUN_TAG=${RAY_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
export RAY_STORE_DIR=${RAY_STORE_DIR:-/dev/shm/opd_exopd_ray_${RAY_RUN_TAG}}
export RAY_SHORT_TMP_DIR=${RAY_SHORT_TMP_DIR:-/tmp/opd_exopd_${RAY_RUN_TAG}}

exec bash "$SCRIPT_DIR/opd_4b.sh" "$@"
