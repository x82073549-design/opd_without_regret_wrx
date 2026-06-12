#!/bin/bash
#SBATCH --job-name=grpo-opd
#SBATCH --output=logs/20251004/grpo_opd_output_%j.log
#SBATCH --error=logs/20251004/grpo_opd_error_%j.log
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --exclude=g[81-82]
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -x

if [ -z "$SLURM_JOB_ID" ]; then
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/grpo_opd_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "Start time: $(date)"
    echo "=========================================="
fi

ray stop --force
export RAY_memory_usage_threshold=0.99
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export PROJECT_NAME=${PROJECT_NAME:-OnPolicyDistillation}
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO

export ADV_ESTIMATOR=token_reward_direct_grpo_gated_opd
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-7168}
export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
export N_RESPONSES=${N_RESPONSES:-4}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
export USE_KL=${USE_KL:-False}
export ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False}
export MODEL_DTYPE=${MODEL_DTYPE:-fp32}
export IS_PLOT=${IS_PLOT:-True}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
export TRAIN_TOTAL_STEPS=${TRAIN_TOTAL_STEPS:-}
export GRPO_OUTCOME_WEIGHT=${GRPO_OUTCOME_WEIGHT:-1.0}

export GRPO_GATED_OPD_ENABLE=${GRPO_GATED_OPD_ENABLE:-True}
export GRPO_GATED_OPD_GATE_MODE=${GRPO_GATED_OPD_GATE_MODE:-reward_positive}
export GRPO_GATED_OPD_CORRECT_THRESHOLD=${GRPO_GATED_OPD_CORRECT_THRESHOLD:-0.5}
export GRPO_GATED_OPD_OPD_COEF=${GRPO_GATED_OPD_OPD_COEF:-1.0}
export GRPO_GATED_OPD_GRPO_COEF=${GRPO_GATED_OPD_GRPO_COEF:-1.0}

if [ "$GRPO_GATED_OPD_ENABLE" != "True" ]; then
    echo "GRPO_GATED_OPD_ENABLE must be True for grpo_opd.sh"
    exit 2
fi
if [ "$LOG_PROB_TOP_K" -le 0 ]; then
    echo "GRPO-Gated OPD requires LOG_PROB_TOP_K > 0"
    exit 2
fi
if [ "$TOP_K_STRATEGY" != "only_stu" ]; then
    echo "GRPO-Gated OPD requires TOP_K_STRATEGY=only_stu"
    exit 2
fi
if [ "$N_RESPONSES" -le 1 ]; then
    echo "GRPO-Gated OPD requires N_RESPONSES > 1"
    exit 2
fi

export TRAIN_DATASET=${TRAIN_DATASET:-datasets/dapo-math-17k.parquet}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-grpo-gated-opd}
export TEST_DATA_DIR=${TEST_DATA_DIR:-datasets/test_data}
TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AIME25/test.parquet", "$TEST_DATA_DIR/AMC23/test.parquet", "$TEST_DATA_DIR/AIME24/test.parquet"]}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-model/DeepSeek-R1-Distill-Qwen-1.5B}
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-model/JustRL-DeepSeek-1.5B}
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

export PROJECT_PATH=${PROJECT_PATH:-checkpoint}
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export TRAIN_TAG=${TRAIN_TAG:-gopd}
RUN_STAMP=$(date +%Y-%m-%d_%H-%M-%S)
export CKPT_PATH=${PROJECT_PATH}/gopd_${TRAIN_TAG}_l${MAX_RESP_LENGTH}_n${N_RESPONSES}_mbs${MINI_BATCH_SIZE}_k${LOG_PROB_TOP_K}_${RUN_STAMP}
echo "GRPO-OPD run config:"
echo "  TRAIN_TAG=$TRAIN_TAG"
echo "  TRAIN_DATASET=$TRAIN_DATASET"
echo "  TRAIN_DATASET_NAME=$TRAIN_DATASET_NAME"
echo "  ACTOR_MODEL_PATH=$ACTOR_MODEL_PATH"
echo "  REWARD_MODEL_PATH=$REWARD_MODEL_PATH"
echo "  MAX_RESP_LENGTH=$MAX_RESP_LENGTH"
echo "  MAX_VAL_RESP_LENGTH=$MAX_VAL_RESP_LENGTH"
echo "  TEMPERATURE=$TEMPERATURE"
echo "  TEACHER_TEMPERATURE=$TEACHER_TEMPERATURE"
echo "  N_RESPONSES=$N_RESPONSES"
echo "  MINI_BATCH_SIZE=$MINI_BATCH_SIZE"
echo "  LOG_PROB_TOP_K=$LOG_PROB_TOP_K"
echo "  TOP_K_STRATEGY=$TOP_K_STRATEGY"
echo "  GRPO_GATED_OPD_GATE_MODE=$GRPO_GATED_OPD_GATE_MODE"
echo "  GRPO_GATED_OPD_CORRECT_THRESHOLD=$GRPO_GATED_OPD_CORRECT_THRESHOLD"
echo "  GRPO_GATED_OPD_OPD_COEF=$GRPO_GATED_OPD_OPD_COEF"
echo "  GRPO_GATED_OPD_GRPO_COEF=$GRPO_GATED_OPD_GRPO_COEF"
echo "  CKPT_PATH=$CKPT_PATH"
OUTLINES_UUID=$(uuidgen 2>/dev/null || python3 -c "import uuid; print(uuid.uuid4())")
export OUTLINES_CACHE_DIR=~/.cache/outlines/${OUTLINES_UUID}
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export SWANLAB_LOG_DIR=${PROJECT_PATH}/swanlab_log
export HYDRA_FULL_ERROR=1
export EXPERIMENT_NAME=$(basename "$CKPT_PATH")

KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

LR_ARGS=""
if [ "${LR_SCHEDULER:-}" = "cosine" ]; then
    LR_ARGS="actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03"
fi

TRAIN_TOTAL_STEPS_ARGS=""
if [ -n "$TRAIN_TOTAL_STEPS" ]; then
    TRAIN_TOTAL_STEPS_ARGS="trainer.total_training_steps=$TRAIN_TOTAL_STEPS"
fi

GRPO_GATED_OPD_ARGS="+algorithm.grpo_gated_opd.enable=$GRPO_GATED_OPD_ENABLE \
+algorithm.grpo_gated_opd.gate_mode=$GRPO_GATED_OPD_GATE_MODE \
+algorithm.grpo_gated_opd.correct_threshold=$GRPO_GATED_OPD_CORRECT_THRESHOLD \
+algorithm.grpo_gated_opd.opd_coef=$GRPO_GATED_OPD_OPD_COEF \
+algorithm.grpo_gated_opd.grpo_coef=$GRPO_GATED_OPD_GRPO_COEF"

PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((1024 + MAX_RESP_LENGTH) > 32768) ? (1024 + MAX_RESP_LENGTH) : 32768))
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"

ray start --head
sleep 5

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.grpo_outcome_weight=$GRPO_OUTCOME_WEIGHT \
    $GRPO_GATED_OPD_ARGS \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$((${MINI_BATCH_SIZE}*${PARALLEL_SIZE})) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    $LR_ARGS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward=$ENABLE_FORMAT_REWARD \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=24 \
    custom_reward_function.path="verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.log_val_generations=2 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=validation_log/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    $TRAIN_TOTAL_STEPS_ARGS \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=$IS_PLOT
MAIN_STATUS=$?

if [ -z "$SLURM_JOB_ID" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi
exit $MAIN_STATUS
