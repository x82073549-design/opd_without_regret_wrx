#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/env.sh" ]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/env.sh"
fi

if [ -f "$ROOT_DIR/.env" ]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
fi

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-model/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-model/JustRL-DeepSeek-1.5B}"
export TRAIN_DATASET="${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}"
export TEST_DATA_DIR="${TEST_DATA_DIR:-datasets/test_data}"
export OPD_CHECK_MIN_GPUS="${OPD_CHECK_MIN_GPUS:-2}"

fail=0

check_path() {
    local path="$1"
    local label="$2"
    if [ -e "$path" ]; then
        echo "[ok] ${label}: ${path}"
    else
        echo "[missing] ${label}: ${path}" >&2
        fail=1
    fi
}

if ! command -v python3 >/dev/null 2>&1; then
    echo "[missing] python3 is not on PATH. Activate the verl-opd conda env first." >&2
    exit 1
fi

if ! python3 - <<'PY'
import importlib
import os
import sys

fail = False

for module_name in ("torch", "ray", "vllm", "verl"):
    try:
        importlib.import_module(module_name)
        print(f"[ok] import {module_name}")
    except Exception as exc:
        print(f"[missing] import {module_name}: {exc}", file=sys.stderr)
        fail = True

try:
    import torch
    print(f"[ok] torch version: {torch.__version__}")
    print(f"[ok] torch CUDA build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("[missing] torch.cuda.is_available() is False", file=sys.stderr)
        fail = True
    else:
        gpu_count = torch.cuda.device_count()
        min_gpus = int(os.environ.get("OPD_CHECK_MIN_GPUS", "2"))
        print(f"[ok] visible CUDA GPUs: {gpu_count}")
        if gpu_count < min_gpus:
            print(
                f"[missing] OPD quickstart expects at least {min_gpus} visible GPUs; "
                f"set CUDA_VISIBLE_DEVICES or OPD_CHECK_MIN_GPUS if needed.",
                file=sys.stderr,
            )
            fail = True
except Exception as exc:
    print(f"[missing] CUDA check failed: {exc}", file=sys.stderr)
    fail = True

if fail:
    sys.exit(1)
PY
then
    fail=1
fi

check_path "$ACTOR_MODEL_PATH" "actor/student model"
check_path "$REWARD_MODEL_PATH" "reward/teacher model"
check_path "$TRAIN_DATASET" "training dataset"
check_path "$TEST_DATA_DIR/AIME24/test.parquet" "AIME24 validation data"
check_path "$TEST_DATA_DIR/AIME25/test.parquet" "AIME25 validation data"
check_path "$TEST_DATA_DIR/AMC23/test.parquet" "AMC23 validation data"

if [ "$fail" -ne 0 ]; then
    echo "Setup check failed. Install missing packages, then edit .env or create symlinks for missing paths." >&2
    exit 1
fi

echo "Setup check passed."
echo "Smoke test: TRAIN_TOTAL_STEPS=1 bash scripts/train/opd_2gpu_80g.sh"
