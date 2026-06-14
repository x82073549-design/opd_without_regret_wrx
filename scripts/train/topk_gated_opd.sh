#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-topk-gated-opd}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}

export TOPK_TOKEN_GATE_OPD_ENABLE=${TOPK_TOKEN_GATE_OPD_ENABLE:-True}
export TOPK_TOKEN_GATE_OPD_BETA=${TOPK_TOKEN_GATE_OPD_BETA:-1.0}
export TOPK_TOKEN_GATE_OPD_CENTER=${TOPK_TOKEN_GATE_OPD_CENTER:-0.0}
export TOPK_TOKEN_GATE_OPD_MIN_GATE=${TOPK_TOKEN_GATE_OPD_MIN_GATE:-0.0}
export TOPK_TOKEN_GATE_OPD_OPD_COEF=${TOPK_TOKEN_GATE_OPD_OPD_COEF:-1.0}
export SAMPLED_TOKEN_GATE_OPD_ENABLE=${SAMPLED_TOKEN_GATE_OPD_ENABLE:-False}

if [ "$ADV_ESTIMATOR" != "token_reward_direct" ]; then
    echo "topk_gated_opd.sh requires ADV_ESTIMATOR=token_reward_direct"
    exit 2
fi
if [ "$LOG_PROB_TOP_K" -le 0 ]; then
    echo "topk_gated_opd.sh requires LOG_PROB_TOP_K > 0"
    exit 2
fi
if [ "$TOP_K_STRATEGY" != "only_stu" ] && [ "$TOP_K_STRATEGY" != "union" ]; then
    echo "topk_gated_opd.sh requires TOP_K_STRATEGY=only_stu or union"
    exit 2
fi
if [ "$TOPK_TOKEN_GATE_OPD_ENABLE" != "True" ]; then
    echo "topk_gated_opd.sh requires TOPK_TOKEN_GATE_OPD_ENABLE=True"
    exit 2
fi
if [ "$SAMPLED_TOKEN_GATE_OPD_ENABLE" = "True" ]; then
    echo "topk_gated_opd.sh cannot be combined with SAMPLED_TOKEN_GATE_OPD_ENABLE=True"
    exit 2
fi

exec bash "$SCRIPT_DIR/on_policy_distillation.sh"
