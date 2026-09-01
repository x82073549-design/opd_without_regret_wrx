#!/usr/bin/env bash

# Wait for the active EOPD resume run to finish, then evaluate the step-279
# PruneOPD, ExOPD, and EOPD checkpoints with scripts/val/eval_opd.sh.
set -Eeuo pipefail

ROOT_DIR=${ROOT_DIR:-/root/opd_without_regret_wrx}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/opd_outputs}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/usr/envs/wrx_env}
GPU_IDS=${GPU_IDS:-0,1}
POLL_SECONDS=${POLL_SECONDS:-30}
GPU_FREE_CONFIRMATIONS=${GPU_FREE_CONFIRMATIONS:-2}

TRAINING_SEED=0
ROOT_STEP=0
DELTA_STEPS=279
N=16
EVAL_SEEDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
MAX_TOKENS=31744
TEMPERATURE=0.7
TOP_P=0.95
THINKING_FLAG=
EVAL_PROTOCOL_TAG=${EVAL_PROTOCOL_TAG:-valfix31744}

PRUNE_RUN_ID=prune_opd_qwen3_4b_non_thinking_rl_math_trainseed0_retry_20260825_092945
EXOPD_RUN_ID=exopd_qwen3_4b_non_thinking_rl_math_trainseed0_step100
EOPD_RUN_ID=eopd_qwen3_4b_non_thinking_rl_math_trainseed0_step100

PRUNE_CKPT=$OUTPUT_ROOT/run_state/$PRUNE_RUN_ID/global_step_279/actor
EXOPD_CKPT=$OUTPUT_ROOT/run_state/$EXOPD_RUN_ID/global_step_279/actor
EOPD_CKPT=$OUTPUT_ROOT/run_state/$EOPD_RUN_ID/global_step_279/actor
TRAINING_LOG=${TRAINING_LOG:-$ROOT_DIR/outputs/logs/orchestration/prune_then_eopd_resume_250_to_279_20260829_systemd_resume250_to279.log}
TRAINING_SUCCESS_MARKER='EOPD step 279 verified; resume chain completed successfully.'
TRAINING_PROCESS_PATTERN='run_prune_then_eopd_resume_250_to_279.sh 0'

MERGED_ROOT=$OUTPUT_ROOT/checkpoint/merged
# Keep corrected generations separate from the invalid earlier run. Mixing
# jsonl files from two protocols in one directory creates duplicate pairs.
EVAL_OUT_ROOT=${EVAL_OUT_ROOT:-$OUTPUT_ROOT/justrl_eval_outputs_$EVAL_PROTOCOL_TAG}
LOG_DIR=${LOG_DIR:-$ROOT_DIR/outputs/logs/eval}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-$LOG_DIR/three_variants_step279_${RUN_TAG}.log}
LOCK_FILE=${LOCK_FILE:-$ROOT_DIR/outputs/three_variants_step279_eval.lock}

mkdir -p "$LOG_DIR" "$MERGED_ROOT" "$EVAL_OUT_ROOT" "$(dirname "$LOCK_FILE")"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another three-variant evaluation waiter is already running: $LOCK_FILE" >&2
    exit 1
fi
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*"
}

on_error() {
    local status=$?
    log "ERROR: evaluation orchestration failed with status $status (line ${BASH_LINENO[0]})."
    exit "$status"
}
trap on_error ERR

validate_positive_integer() {
    local name=$1
    local value=$2
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        log "$name must be a positive integer, got: $value"
        return 1
    fi
}

verify_checkpoint() {
    local actor_dir=$1
    local label=$2
    local step_dir
    step_dir=$(dirname "$actor_dir")
    local required_files=(
        actor/model_world_size_2_rank_0.pt
        actor/model_world_size_2_rank_1.pt
        actor/extra_state_world_size_2_rank_0.pt
        actor/extra_state_world_size_2_rank_1.pt
        actor/fsdp_config.json
        data.pt
    )
    local relative_path
    for relative_path in "${required_files[@]}"; do
        if [ ! -s "$step_dir/$relative_path" ]; then
            log "$label checkpoint is incomplete: $step_dir/$relative_path"
            return 1
        fi
    done
    log "$label checkpoint verified: $actor_dir"
}

training_chain_is_active() {
    pgrep -f -- "$TRAINING_PROCESS_PATTERN" >/dev/null 2>&1
}

training_completed_successfully() {
    [ -f "$TRAINING_LOG" ] && grep -Fq "$TRAINING_SUCCESS_MARKER" "$TRAINING_LOG"
}

wait_for_training() {
    if training_completed_successfully; then
        log "Training success marker already present."
        return
    fi
    if ! training_chain_is_active; then
        log "Training process is absent and the success marker is missing: $TRAINING_LOG"
        return 1
    fi

    log "Waiting for the EOPD resume chain to finish successfully."
    local status_countdown=0
    while training_chain_is_active; do
        if (( status_countdown == 0 )); then
            local latest_step
            latest_step=$(grep -Eo 'training/global_step:[0-9]+' "$TRAINING_LOG" 2>/dev/null | tail -n 1 || true)
            log "EOPD training is still active${latest_step:+ ($latest_step)}."
            status_countdown=20
        fi
        sleep "$POLL_SECONDS"
        status_countdown=$((status_countdown - 1))
    done

    # The checkpoint writer and the parent orchestration shell can exit a few
    # moments apart. Give the final success line a short bounded grace period.
    local grace_checks=10
    while ! training_completed_successfully && (( grace_checks > 0 )); do
        sleep 3
        grace_checks=$((grace_checks - 1))
    done
    if ! training_completed_successfully; then
        log "EOPD training exited without the expected success marker: $TRAINING_LOG"
        return 1
    fi
    log "EOPD training completed successfully."
}

wait_for_free_gpus() {
    local -a gpu_array
    IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
    local free_streak=0
    log "Waiting for GPUs $GPU_IDS to have no compute processes."
    while (( free_streak < GPU_FREE_CONFIRMATIONS )); do
        local -a busy_descriptions=()
        local gpu gpu_processes
        for gpu in "${gpu_array[@]}"; do
            gpu=${gpu//[[:space:]]/}
            if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
                log "Invalid GPU index in GPU_IDS=$GPU_IDS: $gpu"
                return 1
            fi
            if ! gpu_processes=$(nvidia-smi -i "$gpu" \
                --query-compute-apps=pid,process_name,used_memory \
                --format=csv,noheader,nounits 2>&1); then
                log "nvidia-smi failed for GPU $gpu: $gpu_processes"
                return 1
            fi
            if [ -n "${gpu_processes//[[:space:]]/}" ]; then
                busy_descriptions+=("GPU $gpu: $gpu_processes")
            fi
        done

        if (( ${#busy_descriptions[@]} == 0 )); then
            free_streak=$((free_streak + 1))
            log "GPUs $GPU_IDS are free ($free_streak/$GPU_FREE_CONFIRMATIONS confirmations)."
        else
            free_streak=0
            log "GPU compute processes are still present: ${busy_descriptions[*]}"
        fi

        if (( free_streak < GPU_FREE_CONFIRMATIONS )); then
            sleep "$POLL_SECONDS"
        fi
    done
}

run_evaluation() {
    local method=$1
    local checkpoint=$2
    local base_run_id=${method}_trainseed${TRAINING_SEED}_root${ROOT_STEP}_delta${DELTA_STEPS}
    local eval_run_id=${base_run_id}_${EVAL_PROTOCOL_TAG}
    # Reuse the already merged step-279 weights; only the output root and run
    # metadata change for the corrected validation protocol.
    local model_name=${base_run_id}_hf

    verify_checkpoint "$checkpoint" "$method"
    log "Starting $method evaluation through scripts/val/eval_opd.sh."
    ROOT_DIR="$ROOT_DIR" \
    CKPT="$checkpoint" \
    METHOD="$method" \
    TRAINING_SEED="$TRAINING_SEED" \
    ROOT_STEP="$ROOT_STEP" \
    DELTA_STEPS="$DELTA_STEPS" \
    EVAL_RUN_ID="$eval_run_id" \
    MODEL_NAME="$model_name" \
    MERGED_DIR="$MERGED_ROOT/$model_name" \
    OUT_ROOT="$EVAL_OUT_ROOT" \
    GPU_IDS="$GPU_IDS" \
    N="$N" \
    EVAL_SEEDS="$EVAL_SEEDS" \
    MAX_TOKENS="$MAX_TOKENS" \
    TEMPERATURE="$TEMPERATURE" \
    TOP_P="$TOP_P" \
    THINKING_FLAG="$THINKING_FLAG" \
    REPLACE_FLAG= \
    CODE_COMMIT="$CODE_COMMIT" \
    bash "$ROOT_DIR/scripts/val/eval_opd.sh"
    log "$method evaluation completed: $EVAL_OUT_ROOT/$model_name/grading_results.json"
}

validate_positive_integer POLL_SECONDS "$POLL_SECONDS"
validate_positive_integer GPU_FREE_CONFIRMATIONS "$GPU_FREE_CONFIRMATIONS"
if [ ! -x "$CONDA_ENV_PREFIX/bin/python3" ]; then
    log "Python environment is missing: $CONDA_ENV_PREFIX/bin/python3"
    exit 1
fi
if [ ! -f "$ROOT_DIR/scripts/val/eval_opd.sh" ]; then
    log "Required evaluator is missing: $ROOT_DIR/scripts/val/eval_opd.sh"
    exit 1
fi

export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
# env.sh makes the repository's vendored verl package available to the merger.
source "$ROOT_DIR/env.sh"
CODE_COMMIT=$(git -C "$ROOT_DIR" rev-parse HEAD)

log "Three-variant step-279 evaluation waiter started."
log "Config: GPUs=$GPU_IDS tasks=AIME25,AMC23,AIME24 N=$N seeds=$EVAL_SEEDS max_tokens=$MAX_TOKENS temperature=$TEMPERATURE top_p=$TOP_P thinking=model-default grading=rule-based protocol=$EVAL_PROTOCOL_TAG out_root=$EVAL_OUT_ROOT"
log "Evaluation entrypoint: $ROOT_DIR/scripts/val/eval_opd.sh"
log "Log file: $LOG_FILE"

wait_for_training
verify_checkpoint "$EOPD_CKPT" eopd
wait_for_free_gpus

run_evaluation pruneopd "$PRUNE_CKPT"
run_evaluation exopd "$EXOPD_CKPT"
run_evaluation eopd "$EOPD_CKPT"

log "All three evaluations completed successfully."
