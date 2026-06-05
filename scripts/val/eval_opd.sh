#!/bin/bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace/s/ddn/shaojw_group/liujiang/OPD}
CKPT=${CKPT:-}
if [ -z "$CKPT" ]; then
    echo "Usage: CKPT=/path/to/global_step_x/actor bash scripts/val/eval_opd.sh" >&2
    exit 1
fi

STEP_NAME=$(basename "$(dirname "$CKPT")")
MODEL_NAME=${MODEL_NAME:-${STEP_NAME}_hf}
MERGED_DIR=${MERGED_DIR:-${ROOT_DIR}/checkpoint/merged/${MODEL_NAME}}
OUT_ROOT=${OUT_ROOT:-${ROOT_DIR}/justrl_eval_outputs}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
N=${N:-16}
MAX_TOKENS=${MAX_TOKENS:-31744}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
THINKING_FLAG=${THINKING_FLAG:---disable-thinking}
REPLACE_FLAG=${REPLACE_FLAG:-}

cd "$ROOT_DIR"

if [ ! -f "${MERGED_DIR}/config.json" ]; then
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$CKPT" \
        --target_dir "$MERGED_DIR"
else
    echo "Merged HF model already exists at ${MERGED_DIR}; skipping merge."
fi

python3 scripts/val/eval/gen_vllm.py \
    --model "$MERGED_DIR" \
    --out-dir "$OUT_ROOT" \
    --gpus "$GPU_IDS" \
    --n "$N" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --task "AIME25:${ROOT_DIR}/datasets/test_data/AIME25/test.parquet:${N}" \
    --task "AMC23:${ROOT_DIR}/datasets/test_data/AMC23/test.parquet:${N}" \
    --task "AIME24:${ROOT_DIR}/datasets/test_data/AIME24/test.parquet:${N}" \
    $THINKING_FLAG \
    $REPLACE_FLAG

EVAL_DIR="${OUT_ROOT}/$(basename "$MERGED_DIR")"
python3 scripts/val/eval/grade.py \
    --eval-dir "$EVAL_DIR" \
    --output-file "${EVAL_DIR}/grading_results.json"

echo "Evaluation results: ${EVAL_DIR}/grading_results.json"
