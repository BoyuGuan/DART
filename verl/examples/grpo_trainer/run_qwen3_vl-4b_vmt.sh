#!/bin/bash
# GRPO Training Script for Qwen3-VL-4B (Optimized for A100-80GB x4 & Video Data)

set -x

# === 1. 引擎设置 (已写死) ===
# 强制使用 sglang，不再接受参数指定
ENGINE="sglang"

# === 2. 实验名称与后缀处理 ===
# 逻辑：
# 1. 基础名称写死
# 2. 尝试读取第一个参数 $1 作为后缀
# 3. 如果存在后缀，拼接名称并执行 shift (将后缀从参数列表中移除，避免传给 Python 报错)
BASE_EXP_NAME="qwen3_vl_4b_vmt_dart_GRPO"
SUFFIX=$1

if [ -n "$SUFFIX" ]; then
    EXPERIMENT_NAME="${BASE_EXP_NAME}_${SUFFIX}"
    # 【关键修改】移除第一个参数。
    # 这样 $@ 就不会包含 SUFFIX，只包含后面可能的 key=value 参数
    shift
else
    EXPERIMENT_NAME="$BASE_EXP_NAME"
fi

# === 3. 路径配置 ===
TRAIN_DATA=${TRAIN_DATA:-"$HOME/3vmt/data/work3/rl_data/train.parquet"}
VAL_DATA=${VAL_DATA:-"$HOME/3vmt/data/work3/rl_data/val.parquet"}
MODEL_PATH=${MODEL_PATH:-"$HOME/3vmt/qwen3-vl-finetune/output/DART-SFT-Qwen3VL-4B"}
PROJECT_NAME=${PROJECT_NAME:-"verl_grpo_vmt"}

# 输出路径：包含 Experiment Name
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"$HOME/3vmt/checkpoint/DART-RL/${EXPERIMENT_NAME}/$(date +%Y-%m-%d-%H-%M-%S)"}

# === 4. 关键训练参数 (针对视频显存优化) ===
# 1. 总Batch Size
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}

# 2. 序列长度
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

# 3. 学习率与Epoch
LEARNING_RATE=${LEARNING_RATE:-5e-7}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}

# 4. GRPO 采样数
N_ROLLOUTS=${N_ROLLOUTS:-4}

# === 5. GPU 硬件配置 ===
N_GPUS=${N_GPUS:-4}
TP_SIZE=${TP_SIZE:-1}

# === 6. 性能优化参数 ===
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}     
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}    
ROLLOUT_MICRO_BATCH=${ROLLOUT_MICRO_BATCH:-4}      
REF_MICRO_BATCH=${REF_MICRO_BATCH:-4}              
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.8}           

mkdir -p "$CHECKPOINT_DIR"

# === 7. 环境优化 ===
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# === 8. 启动命令 ===
# 注意：最后的 $@ 现在只包含除去后缀之外的参数

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.video_key=video \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_MICRO_BATCH \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=$ENGINE \
    \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_MICRO_BATCH \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    $@