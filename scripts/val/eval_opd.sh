#!/bin/bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace/s/ddn/shaojw_group/liujiang/OPD}
CKPT=${CKPT:-}
if [ -z "$CKPT" ]; then
    echo "Usage: CKPT=/path/to/global_step_x/actor bash scripts/val/eval_opd.sh" >&2
    exit 1
fi

STEP_DIR=$(dirname "$CKPT")
STEP_NAME=$(basename "$STEP_DIR")
EXPERIMENT_DIR=$(dirname "$STEP_DIR")
EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
GLOBAL_STEP=${STEP_NAME#global_step_}
if ! [[ "$GLOBAL_STEP" =~ ^[0-9]+$ ]]; then
    echo "Checkpoint parent directory must be named global_step_<integer>: $STEP_NAME" >&2
    exit 1
fi

METHOD=${METHOD:-$EXPERIMENT_NAME}
TRAINING_SEED=${TRAINING_SEED:-unknown}
ROOT_STEP=${ROOT_STEP:-0}
if ! [[ "$ROOT_STEP" =~ ^[0-9]+$ ]]; then
    echo "ROOT_STEP must be a non-negative integer: $ROOT_STEP" >&2
    exit 1
fi
DELTA_STEPS=${DELTA_STEPS:-$((GLOBAL_STEP - ROOT_STEP))}
if ! [[ "$DELTA_STEPS" =~ ^[0-9]+$ ]]; then
    echo "DELTA_STEPS must be a non-negative integer: $DELTA_STEPS" >&2
    exit 1
fi
if [ "$DELTA_STEPS" -lt 0 ]; then
    echo "DELTA_STEPS must be non-negative: $DELTA_STEPS" >&2
    exit 1
fi

DEFAULT_EVAL_RUN_ID=${METHOD}_trainseed${TRAINING_SEED}_root${ROOT_STEP}_delta${DELTA_STEPS}
EVAL_RUN_ID=${EVAL_RUN_ID:-$DEFAULT_EVAL_RUN_ID}
EVAL_RUN_ID=$(printf '%s' "$EVAL_RUN_ID" | tr -c '[:alnum:]_.-' '_')
MODEL_NAME=${MODEL_NAME:-${EVAL_RUN_ID}_hf}
MERGED_DIR=${MERGED_DIR:-${ROOT_DIR}/checkpoint/merged/${MODEL_NAME}}
OUT_ROOT=${OUT_ROOT:-${ROOT_DIR}/justrl_eval_outputs}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
N=${N:-16}
if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
    echo "N must be a positive integer: $N" >&2
    exit 1
fi
EVAL_SEEDS=${EVAL_SEEDS:-}
if [ -z "$EVAL_SEEDS" ]; then
    EVAL_SEEDS=$(seq -s, 0 $((N - 1)))
fi
SEED_COUNT=$(printf '%s' "$EVAL_SEEDS" | awk -F, '{print NF}')
if [ "$SEED_COUNT" -ne "$N" ]; then
    echo "N=$N must equal the number of EVAL_SEEDS ($SEED_COUNT): $EVAL_SEEDS" >&2
    exit 1
fi
MAX_TOKENS=${MAX_TOKENS:-31744}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
THINKING_FLAG=${THINKING_FLAG:---disable-thinking}
REPLACE_FLAG=${REPLACE_FLAG:-}
VALIDATION_MANIFEST=${VALIDATION_MANIFEST:-}
CODE_COMMIT=${CODE_COMMIT:-}

TASK_ARGS=()
if [ -n "$VALIDATION_MANIFEST" ]; then
    TASK_ARGS=(--task-manifest "$VALIDATION_MANIFEST")
else
    TASK_ARGS=(
        --task "AIME25:${ROOT_DIR}/datasets/test_data/AIME25/test.parquet:${N}"
        --task "AMC23:${ROOT_DIR}/datasets/test_data/AMC23/test.parquet:${N}"
        --task "AIME24:${ROOT_DIR}/datasets/test_data/AIME24/test.parquet:${N}"
    )
fi

cd "$ROOT_DIR"
if [ -z "$CODE_COMMIT" ]; then
    CODE_COMMIT=$(git rev-parse HEAD)
fi

CKPT_ABS=$(cd "$CKPT" && pwd -P)
SOURCE_MARKER="${MERGED_DIR}/.source_checkpoint"
if [ ! -f "${MERGED_DIR}/config.json" ]; then
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$CKPT" \
        --target_dir "$MERGED_DIR"
    printf '%s\n' "$CKPT_ABS" > "$SOURCE_MARKER"
else
    if [ ! -f "$SOURCE_MARKER" ]; then
        echo "Merged model exists without a source checkpoint marker: $MERGED_DIR" >&2
        echo "Choose a new EVAL_RUN_ID or remove the stale merged directory after checking it manually." >&2
        exit 1
    fi
    MERGED_SOURCE=$(cat "$SOURCE_MARKER")
    if [ "$MERGED_SOURCE" != "$CKPT_ABS" ]; then
        echo "EVAL_RUN_ID collision: $MERGED_DIR was created from $MERGED_SOURCE, not $CKPT_ABS" >&2
        exit 1
    fi
    echo "Merged HF model already exists at ${MERGED_DIR}; skipping merge."
fi

python3 scripts/val/eval/gen_vllm.py \
    --model "$MERGED_DIR" \
    --out-dir "$OUT_ROOT" \
    --gpus "$GPU_IDS" \
    --n "$N" \
    --seeds "$EVAL_SEEDS" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    "${TASK_ARGS[@]}" \
    $THINKING_FLAG \
    $REPLACE_FLAG

EVAL_DIR="${OUT_ROOT}/$(basename "$MERGED_DIR")"
python3 scripts/val/eval/grade.py \
    --eval-dir "$EVAL_DIR" \
    --output-file "${EVAL_DIR}/grading_results.json" \
    --detailed-output-file "${EVAL_DIR}/detailed_results.jsonl" \
    --run-id "$EVAL_RUN_ID" \
    --method "$METHOD" \
    --training-seed "$TRAINING_SEED" \
    --root-step "$ROOT_STEP" \
    --delta-steps "$DELTA_STEPS" \
    --code-commit "$CODE_COMMIT"

echo "Evaluation results: ${EVAL_DIR}/grading_results.json"
echo "Detailed results: ${EVAL_DIR}/detailed_results.jsonl"
