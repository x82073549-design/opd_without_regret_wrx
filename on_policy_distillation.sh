#!/bin/bash
# Standalone 2x80G OPD training script. This intentionally does not call
# on_policy_distillation.sh, so 2-GPU settings cannot be overwritten there.
# Usage:
#   cd OPD
#   source env.sh
#   bash scripts/train/opd_2gpu_80g.sh

#SBATCH --job-name=opd-2gpu
#SBATCH --output=logs/20251004/opd_2gpu_%j.out
#SBATCH --error=logs/20251004/opd_2gpu_%j.err
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=${ROOT_DIR:-$SCRIPT_DIR}
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}/verl:${PYTHONPATH:-}"

if [ -f "$ROOT_DIR/env.sh" ]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/env.sh"
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_2gpu_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

if [ "${SKIP_RAY_CLI:-True}" != "True" ]; then
    ray stop --force || true
fi

export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

export PROJECT_NAME=${PROJECT_NAME:-feature_opd_qwen3-1.7B_4B}
export PROJECT_PATH=${PROJECT_PATH:-checkpoint}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${PROJECT_PATH}/swanlab_log}
export WANDB_MODE=${WANDB_MODE:-online}
export TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','wandb']"}

# Algorithm defaults: pure top-k OPD, matching on_policy_distillation.sh behavior.
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export GRPO_OUTCOME_WEIGHT=${GRPO_OUTCOME_WEIGHT:-1.0}

# 2x80G-safe defaults.
export N_GPUS=${N_GPUS:-2}
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-$MAX_RESP_LENGTH}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-16}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$MINI_BATCH_SIZE}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
export N_RESPONSES=${N_RESPONSES:-4}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
export USE_KL=${USE_KL:-False}
export ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export IS_PLOT=${IS_PLOT:-False}
export LOG_OUR_METRICS=${LOG_OUR_METRICS:-True}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
export TRAIN_TOTAL_STEPS=${TRAIN_TOTAL_STEPS:-}
export RESUME_MODE=${RESUME_MODE:-auto}
export RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

# Memory-sensitive knobs for 2x80G.
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.55}
export VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-True}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-8}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export SAVE_FREQ=${SAVE_FREQ:-100}
export TEST_FREQ=${TEST_FREQ:-50}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export VAL_N=${VAL_N:-4}
export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}

export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-opd-2gpu-80g}
export VAL_SOURCE_DATA_DIR=${VAL_SOURCE_DATA_DIR:-scripts/val/data}
export TEST_DATA_DIR=${TEST_DATA_DIR:-scripts/val/data_verl}
export TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME25/test.parquet", "$TEST_DATA_DIR/AMC23/test.parquet", "$TEST_DATA_DIR/AIME24/test.parquet"]}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-/ssd/data/wangruxin/hf_cache/Qwen3-1.7B}
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/ssd/data/wangruxin/hf_cache/hub/Qwen3-4B-Instruct-2507}
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

if [ "${TOKEN_FEATURE_WEIGHT_ENABLE:-False}" = "True" ]; then
    export TOKEN_FEATURE_NAME=${TOKEN_FEATURE_NAME:-teacher_confidence}
    export TOKEN_FEATURE_ALPHA=${TOKEN_FEATURE_ALPHA:-0}
    export TOKEN_FEATURE_DIRECTION=${TOKEN_FEATURE_DIRECTION:-uniform}
    export TOKEN_FEATURE_EPS=${TOKEN_FEATURE_EPS:-1e-6}
    export TOKEN_FEATURE_MIN_WEIGHT=${TOKEN_FEATURE_MIN_WEIGHT:-0.0}
fi

MAX_MODEL_LEN_TRAIN=$((MAX_RESP_LENGTH + MAX_PROMPT_LENGTH))
MAX_MODEL_LEN_VAL=$((MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH))
export MAX_MODEL_LEN=$((MAX_MODEL_LEN_TRAIN > MAX_MODEL_LEN_VAL ? MAX_MODEL_LEN_TRAIN : MAX_MODEL_LEN_VAL))
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( ((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) > 32768) ? (MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) : 32768 ))}

RUN_STAMP=$(date +%Y-%m-%d_%H-%M-%S)
if [ "${TOKEN_FEATURE_WEIGHT_ENABLE:-False}" = "True" ]; then
    DEFAULT_EXPERIMENT_NAME=${TOKEN_FEATURE_NAME}_${TOKEN_FEATURE_DIRECTION}_a${TOKEN_FEATURE_ALPHA}_${RUN_STAMP}
else
    DEFAULT_EXPERIMENT_NAME=baseline_wrx
fi
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}
export CKPT_PATH=${CKPT_PATH:-${PROJECT_PATH}/${EXPERIMENT_NAME}}

KL_ARGS=()
if [ "$USE_KL" = "True" ]; then
    KL_ARGS=(
        actor_rollout_ref.actor.use_kl_loss=True
        actor_rollout_ref.actor.kl_loss_coef=0.005
        actor_rollout_ref.actor.kl_loss_type=low_var_kl
    )
else
    KL_ARGS=(actor_rollout_ref.actor.use_kl_loss=False)
fi

LR_ARGS=()
if [ "${LR_SCHEDULER:-}" = "cosine" ]; then
    LR_ARGS=(
        actor_rollout_ref.actor.optim.warmup_style=cosine
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03
    )
fi

TRAIN_TOTAL_STEPS_ARGS=()
if [ -n "$TRAIN_TOTAL_STEPS" ]; then
    TRAIN_TOTAL_STEPS_ARGS=(trainer.total_training_steps="$TRAIN_TOTAL_STEPS")
fi

RESUME_ARGS=(trainer.resume_mode="$RESUME_MODE")
if [ -n "$RESUME_FROM_PATH" ]; then
    RESUME_ARGS+=(trainer.resume_from_path="$RESUME_FROM_PATH")
fi

TOKEN_FEATURE_WEIGHT_ARGS=()
if [ "${TOKEN_FEATURE_WEIGHT_ENABLE:-False}" = "True" ]; then
    TOKEN_FEATURE_WEIGHT_ARGS=(
        +algorithm.token_feature_weighting.enable=True
        +algorithm.token_feature_weighting.feature="$TOKEN_FEATURE_NAME"
        +algorithm.token_feature_weighting.alpha="$TOKEN_FEATURE_ALPHA"
        +algorithm.token_feature_weighting.direction="$TOKEN_FEATURE_DIRECTION"
        +algorithm.token_feature_weighting.eps="$TOKEN_FEATURE_EPS"
        +algorithm.token_feature_weighting.min_weight="$TOKEN_FEATURE_MIN_WEIGHT"
    )
fi

echo "Run config:"
echo "  SCRIPT=opd_2gpu_80g.sh"
echo "  PROJECT_NAME=$PROJECT_NAME"
echo "  TRAINER_LOGGER=$TRAINER_LOGGER"
echo "  WANDB_ENTITY=${WANDB_ENTITY:-<unset>}"
echo "  N_GPUS=$N_GPUS"
echo "  ADV_ESTIMATOR=$ADV_ESTIMATOR"
echo "  TRAIN_DATASET=$TRAIN_DATASET"
echo "  TRAIN_DATASET_NAME=$TRAIN_DATASET_NAME"
echo "  ACTOR_MODEL_PATH=$ACTOR_MODEL_PATH"
echo "  REWARD_MODEL_PATH=$REWARD_MODEL_PATH"
echo "  MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH"
echo "  MAX_RESP_LENGTH=$MAX_RESP_LENGTH"
echo "  MAX_VAL_RESP_LENGTH=$MAX_VAL_RESP_LENGTH"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "  PPO_MAX_TOKEN_LEN_PER_GPU=$PPO_MAX_TOKEN_LEN_PER_GPU"
echo "  MODEL_DTYPE=$MODEL_DTYPE"
echo "  N_RESPONSES=$N_RESPONSES"
echo "  MINI_BATCH_SIZE=$MINI_BATCH_SIZE"
echo "  TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
echo "  LOG_PROB_TOP_K=$LOG_PROB_TOP_K"
echo "  TOP_K_STRATEGY=$TOP_K_STRATEGY"
echo "  VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION"
echo "  VLLM_ENFORCE_EAGER=$VLLM_ENFORCE_EAGER"
echo "  REWARD_MICRO_BATCH_SIZE_PER_GPU=$REWARD_MICRO_BATCH_SIZE_PER_GPU"
echo "  SAVE_FREQ=$SAVE_FREQ"
echo "  TEST_FREQ=$TEST_FREQ"
echo "  VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN"
echo "  VAL_N=$VAL_N"
echo "  VAL_DO_SAMPLE=$VAL_DO_SAMPLE"
echo "  TEST_DATASET=$TEST_DATASET"
if [ "${TOKEN_FEATURE_WEIGHT_ENABLE:-False}" = "True" ]; then
    echo "  TOKEN_FEATURE_WEIGHT_ENABLE=$TOKEN_FEATURE_WEIGHT_ENABLE"
    echo "  TOKEN_FEATURE_NAME=$TOKEN_FEATURE_NAME"
    echo "  TOKEN_FEATURE_ALPHA=$TOKEN_FEATURE_ALPHA"
    echo "  TOKEN_FEATURE_DIRECTION=$TOKEN_FEATURE_DIRECTION"
    echo "  TOKEN_FEATURE_EPS=$TOKEN_FEATURE_EPS"
    echo "  TOKEN_FEATURE_MIN_WEIGHT=$TOKEN_FEATURE_MIN_WEIGHT"
fi
echo "  CKPT_PATH=$CKPT_PATH"
echo "  EXPERIMENT_NAME=$EXPERIMENT_NAME"

if [ "${SKIP_RAY_CLI:-True}" != "True" ]; then
    ray start --head
    sleep 5
fi

if [ "${PREPARE_VAL_DATA:-True}" = "True" ]; then
    python3 scripts/val/prepare_verl_validation_data.py \
        --source-dir "$VAL_SOURCE_DATA_DIR" \
        --output-dir "$TEST_DATA_DIR" \
        --tasks AIME25 AMC23 AIME24
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$ADV_ESTIMATOR" \
    algorithm.grpo_outcome_weight="$GRPO_OUTCOME_WEIGHT" \
    "${TOKEN_FEATURE_WEIGHT_ARGS[@]}" \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESP_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    "${LR_ARGS[@]}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty="$REPETITION_PENALTY" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward="$ENABLE_FORMAT_REWARD" \
    reward_model.model.path="$REWARD_MODEL_PATH" \
    reward_model.model.input_tokenizer="$ACTOR_MODEL_PATH" \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype="$MODEL_DTYPE" \
    reward_model.micro_batch_size_per_gpu="$REWARD_MICRO_BATCH_SIZE_PER_GPU" \
    custom_reward_function.path=verl/verl/utils/reward_score/ttrl_math/__init__.py \
    custom_reward_function.name=reward_func \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.log_val_generations=2 \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.validation_data_dir="validation_log/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs=1 \
    "${TRAIN_TOTAL_STEPS_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot="$IS_PLOT"

MAIN_STATUS=$?

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi

exit "$MAIN_STATUS"
