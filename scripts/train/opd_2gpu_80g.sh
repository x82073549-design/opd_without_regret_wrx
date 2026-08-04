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
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

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

ray stop --force || true

export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

export PROJECT_NAME=${PROJECT_NAME:-OnPolicyDistillation}
export PROJECT_PATH=${PROJECT_PATH:-checkpoint}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${PROJECT_PATH}/swanlab_log}

# Algorithm defaults: pure top-k OPD, matching on_policy_distillation.sh behavior.
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export GRPO_OUTCOME_WEIGHT=${GRPO_OUTCOME_WEIGHT:-1.0}

# 2x80G-safe defaults.
export N_GPUS=${N_GPUS:-2}
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-7168}
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
export IS_PLOT=${IS_PLOT:-True}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
export METHOD=${METHOD:-opd}
export TRAINING_SEED=${TRAINING_SEED:-0}
export TRAIN_TOTAL_STEPS=${TRAIN_TOTAL_STEPS:-}
export RESUME_MODE=${RESUME_MODE:-disable}
export RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

# Memory-sensitive knobs for 2x80G.
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.55}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-8}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export SAVE_FREQ=${SAVE_FREQ:-20}
export VAL_N=${VAL_N:-16}

# Optional route/gate features, kept compatible with existing code paths.
export OVERLAP_ROUTE_OPD_ENABLE=${OVERLAP_ROUTE_OPD_ENABLE:-False}
export OVERLAP_ROUTE_MODE=${OVERLAP_ROUTE_MODE:-prune_opd}
export OVERLAP_ROUTE_TAU=${OVERLAP_ROUTE_TAU:-0.7}
export OVERLAP_ROUTE_TOP_K=${OVERLAP_ROUTE_TOP_K:-$LOG_PROB_TOP_K}
export OVERLAP_ROUTE_TRIGGER=${OVERLAP_ROUTE_TRIGGER:-first_low}
export OVERLAP_ROUTE_WDROP=${OVERLAP_ROUTE_WDROP:-0.01}
export OVERLAP_ROUTE_WBASE=${OVERLAP_ROUTE_WBASE:-0.5}
export OVERLAP_ROUTE_FKL_COEF=${OVERLAP_ROUTE_FKL_COEF:-0.1}

export SAMPLED_TOKEN_GATE_OPD_ENABLE=${SAMPLED_TOKEN_GATE_OPD_ENABLE:-False}
export SAMPLED_TOKEN_GATE_OPD_BETA=${SAMPLED_TOKEN_GATE_OPD_BETA:-1.0}
export SAMPLED_TOKEN_GATE_OPD_CENTER=${SAMPLED_TOKEN_GATE_OPD_CENTER:-0.0}
export SAMPLED_TOKEN_GATE_OPD_MIN_GATE=${SAMPLED_TOKEN_GATE_OPD_MIN_GATE:-0.0}
export SAMPLED_TOKEN_GATE_OPD_OPD_COEF=${SAMPLED_TOKEN_GATE_OPD_OPD_COEF:-1.0}
export TOPK_TOKEN_GATE_OPD_ENABLE=${TOPK_TOKEN_GATE_OPD_ENABLE:-False}
export TOPK_TOKEN_GATE_OPD_BETA=${TOPK_TOKEN_GATE_OPD_BETA:-1.0}
export TOPK_TOKEN_GATE_OPD_CENTER=${TOPK_TOKEN_GATE_OPD_CENTER:-0.0}
export TOPK_TOKEN_GATE_OPD_MIN_GATE=${TOPK_TOKEN_GATE_OPD_MIN_GATE:-0.0}
export TOPK_TOKEN_GATE_OPD_OPD_COEF=${TOPK_TOKEN_GATE_OPD_OPD_COEF:-1.0}

export OUTCOME_OPD_MASK_ENABLE=${OUTCOME_OPD_MASK_ENABLE:-False}
export OUTCOME_OPD_MASK_CORRECT_THRESHOLD=${OUTCOME_OPD_MASK_CORRECT_THRESHOLD:-0.5}
export OUTCOME_OPD_MASK_WRONG_PREFIX_RATIO=${OUTCOME_OPD_MASK_WRONG_PREFIX_RATIO:-0.25}
export OUTCOME_OPD_MASK_MIN_PREFIX_TOKENS=${OUTCOME_OPD_MASK_MIN_PREFIX_TOKENS:-1}

export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-opd-2gpu-80g}
export TEST_DATA_DIR=${TEST_DATA_DIR:-datasets/test_data}
export TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME25/test.parquet", "$TEST_DATA_DIR/AMC23/test.parquet", "$TEST_DATA_DIR/AIME24/test.parquet"]}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-model/DeepSeek-R1-Distill-Qwen-1.5B}
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-model/JustRL-DeepSeek-1.5B}
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

MAX_MODEL_LEN_TRAIN=$((MAX_RESP_LENGTH + MAX_PROMPT_LENGTH))
MAX_MODEL_LEN_VAL=$((MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH))
export MAX_MODEL_LEN=$((MAX_MODEL_LEN_TRAIN > MAX_MODEL_LEN_VAL ? MAX_MODEL_LEN_TRAIN : MAX_MODEL_LEN_VAL))
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( ((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) > 32768) ? (MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) : 32768 ))}

RUN_STAMP=$(date +%Y-%m-%d_%H-%M-%S)
DEFAULT_EXPERIMENT_NAME=${METHOD}_trainseed${TRAINING_SEED}_${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-tb_${TRAIN_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-2gpu80g-${RUN_STAMP}
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

OVERLAP_ROUTE_ARGS=()
if [ "$OVERLAP_ROUTE_OPD_ENABLE" = "True" ]; then
    OVERLAP_ROUTE_ARGS=(
        +algorithm.overlap_route_opd.enable=True
        +algorithm.overlap_route_opd.mode="$OVERLAP_ROUTE_MODE"
        +algorithm.overlap_route_opd.tau="$OVERLAP_ROUTE_TAU"
        +algorithm.overlap_route_opd.top_k="$OVERLAP_ROUTE_TOP_K"
        +algorithm.overlap_route_opd.trigger="$OVERLAP_ROUTE_TRIGGER"
        +algorithm.overlap_route_opd.wdrop="$OVERLAP_ROUTE_WDROP"
        +algorithm.overlap_route_opd.wbase="$OVERLAP_ROUTE_WBASE"
        +algorithm.overlap_route_opd.fkl_coef="$OVERLAP_ROUTE_FKL_COEF"
    )
fi

SAMPLED_TOKEN_GATE_ARGS=()
if [ "$SAMPLED_TOKEN_GATE_OPD_ENABLE" = "True" ]; then
    SAMPLED_TOKEN_GATE_ARGS=(
        algorithm.sampled_token_gate_opd.enable=True
        algorithm.sampled_token_gate_opd.beta="$SAMPLED_TOKEN_GATE_OPD_BETA"
        algorithm.sampled_token_gate_opd.center="$SAMPLED_TOKEN_GATE_OPD_CENTER"
        algorithm.sampled_token_gate_opd.min_gate="$SAMPLED_TOKEN_GATE_OPD_MIN_GATE"
        algorithm.sampled_token_gate_opd.opd_coef="$SAMPLED_TOKEN_GATE_OPD_OPD_COEF"
    )
fi

TOPK_TOKEN_GATE_ARGS=()
if [ "$TOPK_TOKEN_GATE_OPD_ENABLE" = "True" ]; then
    TOPK_TOKEN_GATE_ARGS=(
        algorithm.topk_token_gate_opd.enable=True
        algorithm.topk_token_gate_opd.beta="$TOPK_TOKEN_GATE_OPD_BETA"
        algorithm.topk_token_gate_opd.center="$TOPK_TOKEN_GATE_OPD_CENTER"
        algorithm.topk_token_gate_opd.min_gate="$TOPK_TOKEN_GATE_OPD_MIN_GATE"
        algorithm.topk_token_gate_opd.opd_coef="$TOPK_TOKEN_GATE_OPD_OPD_COEF"
    )
fi

OUTCOME_OPD_MASK_ARGS=()
if [ "$OUTCOME_OPD_MASK_ENABLE" = "True" ]; then
    OUTCOME_OPD_MASK_ARGS=(
        +algorithm.outcome_opd_mask.enable=True
        +algorithm.outcome_opd_mask.correct_threshold="$OUTCOME_OPD_MASK_CORRECT_THRESHOLD"
        +algorithm.outcome_opd_mask.wrong_prefix_ratio="$OUTCOME_OPD_MASK_WRONG_PREFIX_RATIO"
        +algorithm.outcome_opd_mask.min_prefix_tokens="$OUTCOME_OPD_MASK_MIN_PREFIX_TOKENS"
    )
fi

echo "Run config:"
echo "  SCRIPT=opd_2gpu_80g.sh"
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
echo "  METHOD=$METHOD"
echo "  TRAINING_SEED=$TRAINING_SEED"
echo "  LOG_PROB_TOP_K=$LOG_PROB_TOP_K"
echo "  TOP_K_STRATEGY=$TOP_K_STRATEGY"
echo "  VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION"
echo "  REWARD_MICRO_BATCH_SIZE_PER_GPU=$REWARD_MICRO_BATCH_SIZE_PER_GPU"
echo "  SAVE_FREQ=$SAVE_FREQ"
echo "  SAMPLED_TOKEN_GATE_OPD_ENABLE=$SAMPLED_TOKEN_GATE_OPD_ENABLE"
echo "  TOPK_TOKEN_GATE_OPD_ENABLE=$TOPK_TOKEN_GATE_OPD_ENABLE"
echo "  OUTCOME_OPD_MASK_ENABLE=$OUTCOME_OPD_MASK_ENABLE"
echo "  CKPT_PATH=$CKPT_PATH"
echo "  EXPERIMENT_NAME=$EXPERIMENT_NAME"

ray start --head
sleep 5

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$ADV_ESTIMATOR" \
    algorithm.grpo_outcome_weight="$GRPO_OUTCOME_WEIGHT" \
    "${OVERLAP_ROUTE_ARGS[@]}" \
    "${SAMPLED_TOKEN_GATE_ARGS[@]}" \
    "${TOPK_TOKEN_GATE_ARGS[@]}" \
    "${OUTCOME_OPD_MASK_ARGS[@]}" \
    data.shuffle=False \
    data.seed="$TRAINING_SEED" \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESP_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
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
    +actor_rollout_ref.rollout.seed="$TRAINING_SEED" \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k="$LOG_PROB_TOP_K" \
    +actor_rollout_ref.rollout.top_k_strategy="$TOP_K_STRATEGY" \
    +actor_rollout_ref.rollout.reward_weight_mode="$REWARD_WEIGHT_MODE" \
    +actor_rollout_ref.rollout.teacher_temperature="$TEACHER_TEMPERATURE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$PARALLEL_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.n="$N_RESPONSES" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
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
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype="$MODEL_DTYPE" \
    reward_model.micro_batch_size_per_gpu="$REWARD_MICRO_BATCH_SIZE_PER_GPU" \
    custom_reward_function.path=verl/verl/utils/reward_score/ttrl_math/__init__.py \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.log_val_generations=2 \
    trainer.logger=['console'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.validation_data_dir="validation_log/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq=-1 \
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
