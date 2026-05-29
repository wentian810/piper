#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 多任务 Diffusion Policy 多卡训练脚本
#
# 用法:
#   # 先合并数据集:
#   bash merge_datasets.sh
#
#   # 然后训练（默认 3 卡）:
#   bash train_diffusion_multi_task.sh
#
#   # 单卡训练:
#   NUM_GPUS=1 bash train_diffusion_multi_task.sh
#
#   # 指定 GPU:
#   CUDA_VISIBLE_DEVICES=3,4,5 bash train_diffusion_multi_task.sh
#
# 环境变量（可选覆盖）:
#   DATASET_ROOT=datasets/piper_all       # 合并后数据集路径
#   DATASET_REPO_ID=local/piper_all       # 数据集 repo_id
#   OUTPUT_DIR=outputs/train/piper_multi  # 输出目录
#   JOB_NAME=piper_multi_task             # 任务名
#
#   NUM_GPUS=3                            # 使用的 GPU 数量
#   STEPS=100000
#   BATCH_SIZE=32                         # 每张卡的 batch_size
#   LR=1e-5                               # 学习率
#   WANDB_ENABLE=false
#
#   # Transformer
#   USE_TRANSFORMER=true
#   N_LAYERS=4
#   N_HEADS=8
#   N_EMB=256
#
#   # 多任务
#   NUM_TASKS=4
#   ACTIVE_TASK_ID=0                      # 仅推理时使用
# ============================================================

# ──────────── Dataset / Output ────────────
DATASET_ROOT="${DATASET_ROOT:-/data1/jxyu26/lerobot_piper3/datasets/piper_all}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/piper_all}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/jxyu26/lerobot_piper3/outputs/train/all}"
JOB_NAME="${JOB_NAME:-piper_multi_task}"

# ──────────── Multi-GPU ────────────
NUM_GPUS="${NUM_GPUS:-3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5}"
export CUDA_VISIBLE_DEVICES
MASTER_PORT="${MASTER_PORT:-29500}"

# ──────────── Training Schedule ────────────
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_FREQ="${EVAL_FREQ:--1}"
LOG_FREQ="${LOG_FREQ:-500}"

# ──────────── Hardware / Logging ────────────
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

# ──────────── Policy Core ────────────
N_OBS_STEPS="${N_OBS_STEPS:-8}"
HORIZON="${HORIZON:-128}"
N_ACTION_STEPS="${N_ACTION_STEPS:-100}"
DROP_N_LAST="${DROP_N_LAST:-21}"

# ──────────── Optimizer ────────────
LR="${LR:-1e-5}"

# ──────────── LR Scheduler ────────────
SCHEDULER_NAME="${SCHEDULER_NAME:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"

# ──────────── Transformer ────────────
USE_TRANSFORMER="${USE_TRANSFORMER:-true}"
N_LAYERS="${N_LAYERS:-4}"
N_HEADS="${N_HEADS:-8}"
N_EMB="${N_EMB:-256}"

# ──────────── Multi-task ────────────
NUM_TASKS="${NUM_TASKS:-4}"
ACTIVE_TASK_ID="${ACTIVE_TASK_ID:-0}"

# ═══════════════════════════════════════════════
printf '\n========== 多任务 Diffusion 训练 =========='
printf '\n数据集路径: %s' "$DATASET_ROOT"
printf '\n数据集 repo_id: %s' "$DATASET_REPO_ID"
printf '\n输出目录: %s' "$OUTPUT_DIR"
printf '\njob_name: %s' "$JOB_NAME"
printf '\n--- Policy ---'
printf '\nn_obs_steps: %s' "$N_OBS_STEPS"
printf '\nhorizon: %s' "$HORIZON"
printf '\nn_action_steps: %s' "$N_ACTION_STEPS"
printf '\ndrop_n_last: %s' "$DROP_N_LAST"
printf '\n--- Training ---'
printf '\nsteps: %s' "$STEPS"
printf '\nbatch_size (per GPU): %s' "$BATCH_SIZE"
printf '\neffective batch: %s' "$((BATCH_SIZE * NUM_GPUS))"
printf '\nnum_gpus: %s' "$NUM_GPUS"
printf '\ndevice: %s' "$POLICY_DEVICE"
printf '\n--- Optimizer ---'
printf '\nLR: %s' "$LR"
printf '\n--- Transformer ---'
printf '\nuse_transformer: %s' "$USE_TRANSFORMER"
printf '\nn_layers: %s' "$N_LAYERS"
printf '\nn_heads: %s' "$N_HEADS"
printf '\nn_emb: %s' "$N_EMB"
printf '\n--- Multi-task ---'
printf '\nnum_tasks: %s' "$NUM_TASKS"
printf '\nactive_task_id: %s' "$ACTIVE_TASK_ID"
printf '\n============================================\n\n'

if [ "$NUM_GPUS" -gt 1 ]; then
  accelerate launch \
    --num_processes="${NUM_GPUS}" \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    --main_process_port="${MASTER_PORT}" \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --policy.type=diffusion \
    --policy.use_separate_rgb_encoder_per_camera=true \
    --policy.crop_shape='[480,640]' \
    --policy.n_obs_steps="${N_OBS_STEPS}" \
    --policy.horizon="${HORIZON}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.drop_n_last_frames="${DROP_N_LAST}" \
    --policy.push_to_hub=false \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --steps="${STEPS}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --save_freq="${SAVE_FREQ}" \
    --eval_freq="${EVAL_FREQ}" \
    --policy.device="${POLICY_DEVICE}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --log_freq="${LOG_FREQ}" \
    --policy.use_transformer="${USE_TRANSFORMER}" \
    --policy.n_layers="${N_LAYERS}" \
    --policy.n_heads="${N_HEADS}" \
    --policy.n_emb="${N_EMB}" \
    --policy.num_tasks="${NUM_TASKS}" \
    --policy.active_task_id="${ACTIVE_TASK_ID}" \
    --policy.optimizer_lr="${LR}" \
    --policy.scheduler_name="${SCHEDULER_NAME}" \
    --policy.scheduler_warmup_steps="${WARMUP_STEPS}"
else
  lerobot-train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --policy.type=diffusion \
    --policy.use_separate_rgb_encoder_per_camera=true \
    --policy.crop_shape='[480,640]' \
    --policy.n_obs_steps="${N_OBS_STEPS}" \
    --policy.horizon="${HORIZON}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.drop_n_last_frames="${DROP_N_LAST}" \
    --policy.push_to_hub=false \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --steps="${STEPS}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --save_freq="${SAVE_FREQ}" \
    --eval_freq="${EVAL_FREQ}" \
    --policy.device="${POLICY_DEVICE}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --log_freq="${LOG_FREQ}" \
    --policy.use_transformer="${USE_TRANSFORMER}" \
    --policy.n_layers="${N_LAYERS}" \
    --policy.n_heads="${N_HEADS}" \
    --policy.n_emb="${N_EMB}" \
    --policy.num_tasks="${NUM_TASKS}" \
    --policy.active_task_id="${ACTIVE_TASK_ID}" \
    --policy.optimizer_lr="${LR}" \
    --policy.scheduler_name="${SCHEDULER_NAME}" \
    --policy.scheduler_warmup_steps="${WARMUP_STEPS}"
fi
