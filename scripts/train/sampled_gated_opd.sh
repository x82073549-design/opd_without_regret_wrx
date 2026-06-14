#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-sampled-gated-topk-opd}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export SAMPLED_TOKEN_GATE_OPD_ENABLE=${SAMPLED_TOKEN_GATE_OPD_ENABLE:-True}
export SAMPLED_TOKEN_GATE_OPD_BETA=${SAMPLED_TOKEN_GATE_OPD_BETA:-1.0}
export SAMPLED_TOKEN_GATE_OPD_CENTER=${SAMPLED_TOKEN_GATE_OPD_CENTER:-0.0}
export SAMPLED_TOKEN_GATE_OPD_MIN_GATE=${SAMPLED_TOKEN_GATE_OPD_MIN_GATE:-0.0}
export SAMPLED_TOKEN_GATE_OPD_OPD_COEF=${SAMPLED_TOKEN_GATE_OPD_OPD_COEF:-1.0}

if [ "$LOG_PROB_TOP_K" -le 0 ]; then
    echo "sampled_gated_opd.sh requires LOG_PROB_TOP_K > 0"
    exit 2
fi
if [ "$TOP_K_STRATEGY" != "only_stu" ]; then
    echo "sampled_gated_opd.sh requires TOP_K_STRATEGY=only_stu"
    exit 2
fi
if [ "$SAMPLED_TOKEN_GATE_OPD_ENABLE" != "True" ]; then
    echo "sampled_gated_opd.sh requires SAMPLED_TOKEN_GATE_OPD_ENABLE=True"
    exit 2
fi

exec bash "$SCRIPT_DIR/on_policy_distillation.sh"
