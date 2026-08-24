#!/usr/bin/env bash

# Self-contained local launcher for Qwen3-4B fixed OPD on two A100 80G GPUs.
#
# Usage:
#   bash opd_4b.sh [formal|benchmark|smoke|config] [seed]
#
# Examples:
#   bash opd_4b.sh config 0
#   PHYSICAL_GPUS=5,6 bash opd_4b.sh formal 0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$SCRIPT_DIR
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Local environment and storage
# ---------------------------------------------------------------------------
# These defaults match the current A100 host.  They remain overridable for a
# different conda environment, scheduler allocation, or output volume.
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/ssd/data/wangruxin/envs/wrx_env}
PHYSICAL_GPUS=${PHYSICAL_GPUS:-0,1}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT_DIR/outputs}
HF_HOME=${HF_HOME:-/ssd/data/wangruxin/hf_cache}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}

# Run mode
# ---------------------------------------------------------------------------
RUN_MODE=${1:-formal}
TRAINING_SEED=${2:-0}
BENCHMARK_VLLM_UTILIZATION=${3:-0.50}
BENCHMARK_REWARD_MICRO_BATCH_SIZE=${4:-1}

# ---------------------------------------------------------------------------
# Models and data
# ---------------------------------------------------------------------------
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-/ssd/data/wangruxin/hf_cache/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/ssd/data/wangruxin/hf_cache/models--hbx--JustRL-DeepSeek-1.5B/snapshots/0637e4096c789c67f9eecbe8355e0bdeddede1c2}
TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
VAL_SOURCE_DATA_DIR=${VAL_SOURCE_DATA_DIR:-scripts/val/data}
TEST_DATA_DIR=${TEST_DATA_DIR:-scripts/val/data_verl}
TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME25/test.parquet", "$TEST_DATA_DIR/AMC23/test.parquet", "$TEST_DATA_DIR/AIME24/test.parquet"]}
PREPARE_VAL_DATA=${PREPARE_VAL_DATA:-True}

# ---------------------------------------------------------------------------
# Fixed OPD objective
# ---------------------------------------------------------------------------
METHOD=${METHOD:-fixed_opd}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
GRPO_OUTCOME_WEIGHT=${GRPO_OUTCOME_WEIGHT:-1.0}
LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
USE_KL=${USE_KL:-False}
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-False}
ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}

# ExOPD (G-OPD reward extrapolation). Disabled for the baseline.
G_OPD_ENABLE=${G_OPD_ENABLE:-False}
G_OPD_REWARD_SCALE=${G_OPD_REWARD_SCALE:-1.25}
G_OPD_REFERENCE_MODEL_PATH=${G_OPD_REFERENCE_MODEL_PATH:-$ACTOR_MODEL_PATH}

# Prune-OPD overlap weighting and dynamic response budget. Disabled for the baseline.
PRUNE_OPD_ENABLE=${PRUNE_OPD_ENABLE:-False}
PRUNE_OPD_OVERLAP_THRESHOLD=${PRUNE_OPD_OVERLAP_THRESHOLD:-0.7}
PRUNE_OPD_WDROP=${PRUNE_OPD_WDROP:-0.01}
PRUNE_OPD_WBASE=${PRUNE_OPD_WBASE:-0.5}
PRUNE_OPD_DYNAMIC_LENGTH_ENABLE=${PRUNE_OPD_DYNAMIC_LENGTH_ENABLE:-True}
PRUNE_OPD_DYNAMIC_INITIAL_LENGTH=${PRUNE_OPD_DYNAMIC_INITIAL_LENGTH:-1024}
PRUNE_OPD_DYNAMIC_MIN_LENGTH=${PRUNE_OPD_DYNAMIC_MIN_LENGTH:-1024}
PRUNE_OPD_DYNAMIC_MAX_LENGTH=${PRUNE_OPD_DYNAMIC_MAX_LENGTH:-${MAX_RESP_LENGTH:-7168}}
PRUNE_OPD_DYNAMIC_STEP=${PRUNE_OPD_DYNAMIC_STEP:-100}
PRUNE_OPD_DYNAMIC_MARGIN=${PRUNE_OPD_DYNAMIC_MARGIN:-100}
PRUNE_OPD_DYNAMIC_HIT_RATIO=${PRUNE_OPD_DYNAMIC_HIT_RATIO:-0.1}
PRUNE_OPD_DYNAMIC_SHRINK_PATIENCE=${PRUNE_OPD_DYNAMIC_SHRINK_PATIENCE:-3}
PRUNE_OPD_DYNAMIC_EPSILON=${PRUNE_OPD_DYNAMIC_EPSILON:-1e-6}

# ---------------------------------------------------------------------------
# Sampling, batching, and optimization
# ---------------------------------------------------------------------------
N_GPUS=${N_GPUS:-2}
PARALLEL_SIZE=${PARALLEL_SIZE:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-16348}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-1}
N_RESPONSES=${N_RESPONSES:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
# An explicitly empty value suppresses this optional chat-template argument.
ENABLE_THINKING=${ENABLE_THINKING-False}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.50}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-True}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
ACTOR_PPO_EPOCHS=${ACTOR_PPO_EPOCHS:-1}

# ---------------------------------------------------------------------------
# Training and validation schedule
# ---------------------------------------------------------------------------

# One pass over the 17K training set is 279 optimizer steps with the
# baseline batch setting.  Validation runs only when that pass completes.
TRAIN_TOTAL_STEPS=${TRAIN_TOTAL_STEPS:-279}
TRAIN_TOTAL_EPOCHS=${TRAIN_TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-279}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
VAL_ONLY=${VAL_ONLY:-False}
VAL_N=${VAL_N:-4}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7}
VAL_TOP_P=${VAL_TOP_P:-0.95}
DATA_SHUFFLE=${DATA_SHUFFLE:-False}
VALIDATION_SHUFFLE=${VALIDATION_SHUFFLE:-False}
RESUME_MODE=${RESUME_MODE:-disable}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

# ---------------------------------------------------------------------------
# Runtime and logging
# ---------------------------------------------------------------------------
PROJECT_NAME=${PROJECT_NAME:-opd_qwen3_4b}
TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','wandb']"}
WANDB_MODE=${WANDB_MODE:-offline}
IS_PLOT=${IS_PLOT:-False}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-24}
GPU_NAME_PATTERN=${GPU_NAME_PATTERN:-A100}
GPU_MIN_MEMORY_MIB=${GPU_MIN_MEMORY_MIB:-80000}
ACTOR_USE_REMOVE_PADDING=${ACTOR_USE_REMOVE_PADDING:-True}
REWARD_USE_REMOVE_PADDING=${REWARD_USE_REMOVE_PADDING:-True}
CHECKPOINT_SAVE_CONTENTS=${CHECKPOINT_SAVE_CONTENTS:-model,optimizer,extra}

case "$RUN_MODE" in
    formal)
        RUN_SUFFIX=step100
        ;;
    benchmark)
        RUN_SUFFIX=benchmark_step1_$(date +%Y%m%d_%H%M%S)
        TRAIN_TOTAL_STEPS=1
        VAL_BEFORE_TRAIN=False
        TEST_FREQ=0
        SAVE_FREQ=0
        TRAINER_LOGGER="['console']"
        WANDB_MODE=disabled
        VLLM_GPU_MEMORY_UTILIZATION=$BENCHMARK_VLLM_UTILIZATION
        REWARD_MICRO_BATCH_SIZE_PER_GPU=$BENCHMARK_REWARD_MICRO_BATCH_SIZE
        ;;
    smoke)
        RUN_SUFFIX=smoke1_$(date +%Y%m%d_%H%M%S)
        TRAIN_TOTAL_STEPS=1
        MAX_RESP_LENGTH=512
        MAX_VAL_RESP_LENGTH=512
        VAL_BEFORE_TRAIN=False
        TEST_FREQ=0
        SAVE_FREQ=0
        TRAINER_LOGGER="['console']"
        WANDB_MODE=offline
        ;;
    config)
        RUN_SUFFIX=config_$(date +%Y%m%d_%H%M%S)
        TRAINER_LOGGER="['console']"
        WANDB_MODE=offline
        ;;
    *)
        echo "Usage: $0 [formal|benchmark|smoke|config] [seed]" >&2
        exit 2
        ;;
esac

# Smoke mode shortens the shared baseline response budget; keep the Prune-OPD
# controller inside that reduced capacity as well.
if [ "$PRUNE_OPD_ENABLE" = True ] && [ "$PRUNE_OPD_DYNAMIC_MAX_LENGTH" -gt "$MAX_RESP_LENGTH" ]; then
    PRUNE_OPD_DYNAMIC_MAX_LENGTH=$MAX_RESP_LENGTH
fi
if [ "$PRUNE_OPD_ENABLE" = True ] && [ "$PRUNE_OPD_DYNAMIC_INITIAL_LENGTH" -gt "$PRUNE_OPD_DYNAMIC_MAX_LENGTH" ]; then
    PRUNE_OPD_DYNAMIC_INITIAL_LENGTH=$PRUNE_OPD_DYNAMIC_MAX_LENGTH
fi
if [ "$PRUNE_OPD_ENABLE" = True ] && [ "$PRUNE_OPD_DYNAMIC_MIN_LENGTH" -gt "$PRUNE_OPD_DYNAMIC_INITIAL_LENGTH" ]; then
    PRUNE_OPD_DYNAMIC_MIN_LENGTH=$PRUNE_OPD_DYNAMIC_INITIAL_LENGTH
fi

if [ ! -x "$CONDA_ENV_PREFIX/bin/python3" ]; then
    echo "Missing Python environment: $CONDA_ENV_PREFIX" >&2
    exit 2
fi
export CONDA_PREFIX=$CONDA_ENV_PREFIX
export CONDA_DEFAULT_ENV=opd_env
export PATH=$CONDA_ENV_PREFIX/bin:$PATH
export HF_HOME HF_DATASETS_CACHE
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"
export PYTHONHASHSEED=$TRAINING_SEED
export CUDA_VISIBLE_DEVICES=$PHYSICAL_GPUS
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export OPD_DETERMINISTIC_TRAINING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE

if ! [[ "$TRAINING_SEED" =~ ^[0-9]+$ ]]; then
    echo "Seed must be a non-negative integer: $TRAINING_SEED" >&2
    exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "$PHYSICAL_GPUS"
if [ "${#GPU_IDS[@]}" -ne 2 ]; then
    echo "PHYSICAL_GPUS must contain exactly two GPU indices: $PHYSICAL_GPUS" >&2
    exit 2
fi

python3 -c 'import numpy, torch, verl' >/dev/null
for required_path in "$ACTOR_MODEL_PATH/config.json" "$REWARD_MODEL_PATH/config.json" "$TRAIN_DATASET"; do
    if [ ! -f "$required_path" ]; then
        echo "Missing required artifact: $required_path" >&2
        exit 2
    fi
done

# Keep validation prompts consistent with the OPD math evaluation protocol.
# This writes prepared copies under TEST_DATA_DIR and leaves the source data
# under VAL_SOURCE_DATA_DIR unchanged.
if [ "$PREPARE_VAL_DATA" = True ]; then
    python3 scripts/val/prepare_verl_validation_data.py \
        --source-dir "$VAL_SOURCE_DATA_DIR" \
        --output-dir "$TEST_DATA_DIR" \
        --tasks AIME25 AMC23 AIME24
fi
for required_path in \
    "$TEST_DATA_DIR/AIME25/test.parquet" \
    "$TEST_DATA_DIR/AMC23/test.parquet" \
    "$TEST_DATA_DIR/AIME24/test.parquet"; do
    if [ ! -f "$required_path" ]; then
        echo "Missing required validation artifact: $required_path" >&2
        exit 2
    fi
done

if [ "$RUN_MODE" != config ]; then
    for gpu in "${GPU_IDS[@]}"; do
        if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
            echo "Invalid GPU index: $gpu" >&2
            exit 2
        fi
        gpu_name=$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader | tr -d '\r')
        gpu_mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')
        if [[ "$gpu_name" != *"$GPU_NAME_PATTERN"* ]] || [ "$gpu_mem" -lt "$GPU_MIN_MEMORY_MIB" ]; then
            echo "GPU $gpu does not satisfy the ${GPU_NAME_PATTERN} ${GPU_MIN_MEMORY_MIB}MiB requirement: name=$gpu_name memory=${gpu_mem}MiB" >&2
            exit 2
        fi
    done
fi

MAX_MODEL_LEN_TRAIN=$((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH))
MAX_MODEL_LEN_VAL=$((MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH))
MAX_MODEL_LEN=$((MAX_MODEL_LEN_TRAIN > MAX_MODEL_LEN_VAL ? MAX_MODEL_LEN_TRAIN : MAX_MODEL_LEN_VAL))
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_MODEL_LEN_TRAIN > 32768 ? MAX_MODEL_LEN_TRAIN : 32768))}

RUN_ID=${RUN_ID:-${METHOD}_qwen3_4b_non_thinking_rl_math_trainseed${TRAINING_SEED}_${RUN_SUFFIX}}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$RUN_ID}
if [ "$RUN_MODE" = formal ]; then
    RUN_ROOT=$OUTPUT_ROOT
else
    RUN_ROOT=$OUTPUT_ROOT/preflight
fi
PROJECT_PATH=${PROJECT_PATH:-$RUN_ROOT/run_state}
CKPT_PATH=${CKPT_PATH:-$PROJECT_PATH/$RUN_ID}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-$RUN_ROOT/validation/$RUN_ID}
LOG_ROOT=${LOG_ROOT:-$RUN_ROOT/logs/internal}
LOG_DIR=${LOG_DIR:-$LOG_ROOT/$RUN_ID}
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$RUN_ROOT/swanlab}
WANDB_DIR=${WANDB_DIR:-$RUN_ROOT/wandb}
WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$RUN_ROOT/wandb_cache}
WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-$RUN_ROOT/wandb_config}
WANDB_ARTIFACT_DIR=${WANDB_ARTIFACT_DIR:-$RUN_ROOT/wandb_artifacts}
OUTLINES_CACHE_DIR=${OUTLINES_CACHE_DIR:-$RUN_ROOT/outlines_cache/$RUN_ID}
RAY_LOG_ROOT=${RAY_LOG_ROOT:-$RUN_ROOT/logs/ray}
RAY_RUN_STAMP=$(date +%Y%m%d_%H%M%S)
RAY_STORE_DIR=${RAY_STORE_DIR:-$RAY_LOG_ROOT/ray_store_${RAY_RUN_STAMP}}
RAY_SHORT_TMP_DIR=${RAY_SHORT_TMP_DIR:-/tmp/opd_${RAY_RUN_STAMP}}
if [ -e "$RAY_SHORT_TMP_DIR" ] && [ ! -L "$RAY_SHORT_TMP_DIR" ]; then
    echo "RAY_SHORT_TMP_DIR exists and is not a symlink: $RAY_SHORT_TMP_DIR" >&2
    exit 2
fi
mkdir -p "$RAY_STORE_DIR"
ln -sfn "$RAY_STORE_DIR" "$RAY_SHORT_TMP_DIR"
RAY_TMP_DIR=${RAY_TMP_DIR:-$RAY_SHORT_TMP_DIR}
TMPDIR=${TMPDIR:-$RAY_TMP_DIR/tmp}
export PROJECT_PATH CKPT_PATH VALIDATION_DATA_DIR LOG_ROOT LOG_DIR SWANLAB_LOG_DIR
export WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_ARTIFACT_DIR
export OUTLINES_CACHE_DIR RAY_TMP_DIR TMPDIR

if [ -e "$CKPT_PATH" ]; then
    echo "Refusing to reuse existing run directory: $CKPT_PATH" >&2
    exit 2
fi
mkdir -p "$CKPT_PATH" "$VALIDATION_DATA_DIR" "$LOG_DIR" "$SWANLAB_LOG_DIR" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_ARTIFACT_DIR" \
    "$OUTLINES_CACHE_DIR" "$RAY_TMP_DIR" "$TMPDIR"

LOG_FILE=$LOG_DIR/run_2gpu_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Qwen3-4B OPD configuration:"
echo "  mode=$RUN_MODE seed=$TRAINING_SEED GPUs=$PHYSICAL_GPUS"
echo "  method=$METHOD g_opd=$G_OPD_ENABLE prune_opd=$PRUNE_OPD_ENABLE"
echo "  actor=$ACTOR_MODEL_PATH"
echo "  teacher=$REWARD_MODEL_PATH"
echo "  train_batch=$TRAIN_BATCH_SIZE mini_batch=$MINI_BATCH_SIZE responses=$N_RESPONSES"
echo "  prompt=$MAX_PROMPT_LENGTH response=$MAX_RESP_LENGTH val_response=$MAX_VAL_RESP_LENGTH"
echo "  total_steps=$TRAIN_TOTAL_STEPS save_freq=$SAVE_FREQ test_freq=$TEST_FREQ"
echo "  checkpoint=$CKPT_PATH"
echo "  log=$LOG_FILE"

CONFIG_ONLY_ARGS=()
if [ "$RUN_MODE" = config ]; then
    CONFIG_ONLY_ARGS=(--cfg job)
fi
KL_ARGS=(actor_rollout_ref.actor.use_kl_loss=False)
if [ "$USE_KL" = True ]; then
    KL_ARGS=(actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.005 actor_rollout_ref.actor.kl_loss_type=low_var_kl)
fi
RESUME_ARGS=(trainer.resume_mode="$RESUME_MODE")
if [ -n "$RESUME_FROM_PATH" ]; then
    RESUME_ARGS+=(trainer.resume_from_path="$RESUME_FROM_PATH")
fi
if [ "$G_OPD_ENABLE" = True ] && [ "$PRUNE_OPD_ENABLE" = True ]; then
    echo "G_OPD_ENABLE and PRUNE_OPD_ENABLE are mutually exclusive reproduction variants" >&2
    exit 2
fi
G_OPD_ARGS=()
if [ "$G_OPD_ENABLE" = True ]; then
    G_OPD_ARGS=(
        +algorithm.g_opd.enable=True
        +algorithm.g_opd.reward_scale="$G_OPD_REWARD_SCALE"
        +actor_rollout_ref.ref.model.path="$G_OPD_REFERENCE_MODEL_PATH"
    )
fi
PRUNE_OPD_ARGS=()
if [ "$PRUNE_OPD_ENABLE" = True ]; then
    PRUNE_OPD_ARGS=(
        +algorithm.overlap_route_opd.enable=True
        +algorithm.overlap_route_opd.mode=prune_opd
        +algorithm.overlap_route_opd.top_k="$LOG_PROB_TOP_K"
        +algorithm.overlap_route_opd.trigger=first_low
        +algorithm.overlap_route_opd.tau="$PRUNE_OPD_OVERLAP_THRESHOLD"
        +algorithm.overlap_route_opd.wdrop="$PRUNE_OPD_WDROP"
        +algorithm.overlap_route_opd.wbase="$PRUNE_OPD_WBASE"
        +algorithm.overlap_route_opd.dynamic_length.enable="$PRUNE_OPD_DYNAMIC_LENGTH_ENABLE"
        +algorithm.overlap_route_opd.dynamic_length.initial="$PRUNE_OPD_DYNAMIC_INITIAL_LENGTH"
        +algorithm.overlap_route_opd.dynamic_length.min="$PRUNE_OPD_DYNAMIC_MIN_LENGTH"
        +algorithm.overlap_route_opd.dynamic_length.max="$PRUNE_OPD_DYNAMIC_MAX_LENGTH"
        +algorithm.overlap_route_opd.dynamic_length.step="$PRUNE_OPD_DYNAMIC_STEP"
        +algorithm.overlap_route_opd.dynamic_length.margin="$PRUNE_OPD_DYNAMIC_MARGIN"
        +algorithm.overlap_route_opd.dynamic_length.hit_ratio_threshold="$PRUNE_OPD_DYNAMIC_HIT_RATIO"
        +algorithm.overlap_route_opd.dynamic_length.shrink_patience="$PRUNE_OPD_DYNAMIC_SHRINK_PATIENCE"
        +algorithm.overlap_route_opd.dynamic_length.epsilon="$PRUNE_OPD_DYNAMIC_EPSILON"
    )
fi
CHAT_TEMPLATE_ARGS=()
if [ -n "$ENABLE_THINKING" ]; then
    CHAT_TEMPLATE_ARGS=(+data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING")
fi

set -x
set +e
python3 -m verl.trainer.main_ppo \
    "${CONFIG_ONLY_ARGS[@]}" \
    algorithm.adv_estimator="$ADV_ESTIMATOR" \
    algorithm.grpo_outcome_weight="$GRPO_OUTCOME_WEIGHT" \
    algorithm.use_kl_in_reward="$USE_KL_IN_REWARD" \
    "${G_OPD_ARGS[@]}" \
    "${PRUNE_OPD_ARGS[@]}" \
    data.shuffle="$DATA_SHUFFLE" \
    data.validation_shuffle="$VALIDATION_SHUFFLE" \
    data.seed="$TRAINING_SEED" \
    data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESP_LENGTH" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding="$ACTOR_USE_REMOVE_PADDING" \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_epochs="$ACTOR_PPO_EPOCHS" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$PARALLEL_SIZE" \
    "${KL_ARGS[@]}" \
    actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype="$MODEL_DTYPE" \
    actor_rollout_ref.actor.checkpoint.save_contents="[$CHECKPOINT_SAVE_CONTENTS]" \
    actor_rollout_ref.actor.checkpoint.load_contents="[$CHECKPOINT_SAVE_CONTENTS]" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype="$MODEL_DTYPE" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k="$LOG_PROB_TOP_K" \
    +actor_rollout_ref.rollout.top_k_strategy="$TOP_K_STRATEGY" \
    +actor_rollout_ref.rollout.reward_weight_mode="$REWARD_WEIGHT_MODE" \
    +actor_rollout_ref.rollout.teacher_temperature="$TEACHER_TEMPERATURE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$PARALLEL_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.enforce_eager="$VLLM_ENFORCE_EAGER" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.n="$N_RESPONSES" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE" \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens="$MAX_VAL_RESP_LENGTH" \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
    actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P" \
    actor_rollout_ref.rollout.repetition_penalty="$REPETITION_PENALTY" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward="$ENABLE_FORMAT_REWARD" \
    reward_model.model.path="$REWARD_MODEL_PATH" \
    reward_model.model.input_tokenizer="$ACTOR_MODEL_PATH" \
    reward_model.model.use_remove_padding="$REWARD_USE_REMOVE_PADDING" \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype="$MODEL_DTYPE" \
    reward_model.micro_batch_size_per_gpu="$REWARD_MICRO_BATCH_SIZE_PER_GPU" \
    custom_reward_function.path=verl/verl/utils/reward_score/ttrl_math/__init__.py \
    custom_reward_function.name=reward_func \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.val_only="$VAL_ONLY" \
    trainer.log_val_generations=2 \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    +trainer.seed="$TRAINING_SEED" \
    trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TRAIN_TOTAL_EPOCHS" \
    trainer.total_training_steps="$TRAIN_TOTAL_STEPS" \
    "${RESUME_ARGS[@]}" \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot="$IS_PLOT" \
    ray_kwargs.ray_init.num_cpus="$RAY_NUM_CPUS" \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMP_DIR"
MAIN_STATUS=$?
set -e

echo "End time: $(date)"
exit "$MAIN_STATUS"
