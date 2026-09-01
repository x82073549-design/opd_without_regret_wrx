#!/usr/bin/env bash

# Resume the existing Prune-OPD and EOPD runs at step 250, serially, and
# produce step-279 checkpoints without running validation.
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

SEED=${1:-0}
PHYSICAL_GPUS=${PHYSICAL_GPUS:-0,1}
WANDB_MODE=${WANDB_MODE:-offline}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/opd_outputs}
CHAIN_TAG=${CHAIN_TAG:-$(date +%Y%m%d_%H%M%S)}
CHAIN_LOG_DIR=${CHAIN_LOG_DIR:-$SCRIPT_DIR/outputs/logs/orchestration}
CHAIN_LOG=${CHAIN_LOG:-$CHAIN_LOG_DIR/prune_then_eopd_resume_250_to_279_${CHAIN_TAG}.log}

PRUNE_RUN_ID=${PRUNE_RUN_ID:-prune_opd_qwen3_4b_non_thinking_rl_math_trainseed${SEED}_retry_20260825_092945}
EOPD_RUN_ID=${EOPD_RUN_ID:-eopd_qwen3_4b_non_thinking_rl_math_trainseed${SEED}_step100}
PRUNE_STEP_250=${PRUNE_STEP_250:-$OUTPUT_ROOT/run_state/$PRUNE_RUN_ID/global_step_250}
EOPD_STEP_250=${EOPD_STEP_250:-$OUTPUT_ROOT/run_state/$EOPD_RUN_ID/global_step_250}
PRUNE_STEP_279=$OUTPUT_ROOT/run_state/$PRUNE_RUN_ID/global_step_279
EOPD_STEP_279=$OUTPUT_ROOT/run_state/$EOPD_RUN_ID/global_step_279

mkdir -p "$CHAIN_LOG_DIR"
exec > >(tee -a "$CHAIN_LOG") 2>&1

export PHYSICAL_GPUS WANDB_MODE OUTPUT_ROOT
# The host has 31 GiB physical RAM and a 64 GiB swapfile. Disable Ray's
# physical-memory kill threshold so the kernel can spill cold pages to swap.
export RAY_memory_monitor_refresh_ms=0

verify_common_checkpoint() {
    local checkpoint=$1
    local label=$2

    local required_files=(
        actor/model_world_size_2_rank_0.pt
        actor/model_world_size_2_rank_1.pt
        actor/extra_state_world_size_2_rank_0.pt
        actor/extra_state_world_size_2_rank_1.pt
        data.pt
    )
    local relative_path
    for relative_path in "${required_files[@]}"; do
        if [ ! -s "$checkpoint/$relative_path" ]; then
            echo "[$(date)] $label checkpoint is incomplete: $checkpoint/$relative_path" >&2
            return 1
        fi
    done
}

verify_common_checkpoint "$PRUNE_STEP_250" "Prune-OPD input"
if [ ! -s "$PRUNE_STEP_250/prune_opd_length_controller.pt" ]; then
    echo "[$(date)] Prune-OPD controller state is missing: $PRUNE_STEP_250/prune_opd_length_controller.pt" >&2
    exit 1
fi
verify_common_checkpoint "$EOPD_STEP_250" "EOPD input"

echo "[$(date)] resume chain started"
echo "  GPUs=$PHYSICAL_GPUS seed=$SEED wandb=$WANDB_MODE"
echo "  validation=disabled (test_freq=0, val_before_train=False)"
echo "  prune=$PRUNE_STEP_250 -> $PRUNE_STEP_279"
echo "  eopd=$EOPD_STEP_250 -> $EOPD_STEP_279"
echo "  chain_log=$CHAIN_LOG"

RUN_ID="$PRUNE_RUN_ID" \
RESUME_MODE=resume_path \
RESUME_FROM_PATH="$PRUNE_STEP_250" \
TRAIN_TOTAL_STEPS=279 \
SAVE_FREQ=50 \
TEST_FREQ=0 \
VAL_BEFORE_TRAIN=False \
PREPARE_VAL_DATA=False \
bash "$SCRIPT_DIR/prune_opd_4b.sh" formal "$SEED"

verify_common_checkpoint "$PRUNE_STEP_279" "Prune-OPD output"
if [ ! -s "$PRUNE_STEP_279/prune_opd_length_controller.pt" ]; then
    echo "[$(date)] Prune-OPD step-279 controller state is missing" >&2
    exit 1
fi
echo "[$(date)] Prune-OPD step 279 verified; starting EOPD."

RUN_ID="$EOPD_RUN_ID" \
RESUME_MODE=resume_path \
RESUME_FROM_PATH="$EOPD_STEP_250" \
TRAIN_TOTAL_STEPS=279 \
SAVE_FREQ=50 \
TEST_FREQ=0 \
VAL_BEFORE_TRAIN=False \
PREPARE_VAL_DATA=False \
bash "$SCRIPT_DIR/eopd_4b.sh" formal "$SEED"

verify_common_checkpoint "$EOPD_STEP_279" "EOPD output"
echo "[$(date)] EOPD step 279 verified; resume chain completed successfully."
