#!/usr/bin/env bash

# Run Prune-OPD and start ExOPD only after Prune-OPD exits successfully.
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

SEED=${1:-0}
PHYSICAL_GPUS=${PHYSICAL_GPUS:-0,1}
WANDB_MODE=${WANDB_MODE:-offline}
CHAIN_TAG=${CHAIN_TAG:-$(date +%Y%m%d_%H%M%S)}
CHAIN_LOG_DIR=${CHAIN_LOG_DIR:-$SCRIPT_DIR/outputs/logs/orchestration}
CHAIN_LOG=${CHAIN_LOG:-$CHAIN_LOG_DIR/prune_then_exopd_${CHAIN_TAG}.log}

mkdir -p "$CHAIN_LOG_DIR"
exec > >(tee -a "$CHAIN_LOG") 2>&1

export PHYSICAL_GPUS WANDB_MODE

PRUNE_RUN_ID=${PRUNE_RUN_ID:-prune_opd_qwen3_4b_non_thinking_rl_math_trainseed${SEED}_retry_${CHAIN_TAG}}
EXOPD_RUN_ID=${EXOPD_RUN_ID:-exopd_qwen3_4b_non_thinking_rl_math_trainseed${SEED}_after_prune_${CHAIN_TAG}}

echo "[$(date)] chain started"
echo "  GPUs=$PHYSICAL_GPUS seed=$SEED wandb=$WANDB_MODE"
echo "  prune_run_id=$PRUNE_RUN_ID"
echo "  exopd_run_id=$EXOPD_RUN_ID"
echo "  chain_log=$CHAIN_LOG"

set +e
RUN_ID="$PRUNE_RUN_ID" bash "$SCRIPT_DIR/prune_opd_4b.sh" formal "$SEED"
PRUNE_STATUS=$?
set -e

if [ "$PRUNE_STATUS" -ne 0 ]; then
    echo "[$(date)] Prune-OPD failed with status $PRUNE_STATUS; ExOPD will not start."
    exit "$PRUNE_STATUS"
fi

echo "[$(date)] Prune-OPD completed successfully; starting ExOPD."

set +e
RUN_ID="$EXOPD_RUN_ID" bash "$SCRIPT_DIR/exopd_4b.sh" formal "$SEED"
EXOPD_STATUS=$?
set -e

if [ "$EXOPD_STATUS" -ne 0 ]; then
    echo "[$(date)] ExOPD failed with status $EXOPD_STATUS."
    exit "$EXOPD_STATUS"
fi

echo "[$(date)] ExOPD completed successfully; chain finished."
