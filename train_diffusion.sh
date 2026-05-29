#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash train_diffusion.sh
#
# Optional environment variables:
#   DATASET_ROOT=/home/bbncf305/lerobot513/DP/lerobot_piper3/datasets/piper1
#   DATASET_REPO_ID=local/piper1
#   OUTPUT_DIR=outputs/train/diffusion_piper1
#   JOB_NAME=diffusion_piper1
#
#   STEPS=50000
#   BATCH_SIZE=16
#   NUM_WORKERS=4
#   SAVE_FREQ=10000
#   EVAL_FREQ=1000
#   POLICY_DEVICE=cuda
#   WANDB_ENABLE=false
#
#   # ====== Optimizer & Scheduler (Diffusion Policy Presets) ======
#   LR=1e-4
#   OPTIM_BETAS="(0.95, 0.999)"
#   OPTIM_EPS=1e-8
#   WEIGHT_DECAY=1e-6
#   SCHEDULER_NAME=cosine
#   WARMUP_STEPS=500
#   GRAD_CLIP_NORM=10.0

# ──────────── Dataset / Output ────────────
DATASET_ROOT="${DATASET_ROOT:-/home/bbncf305/lerobot513/DP/lerobot_piper3/datasets/piper3}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/piper3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/diffusion_piper3}"
JOB_NAME="${JOB_NAME:-diffusion_piper3}"

# ──────────── Training Schedule ────────────
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_FREQ="${EVAL_FREQ:-1000}"

# ──────────── Hardware / Logging ────────────
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

# ──────────── Optimizer ────────────
LR="${LR:-1e-5}"
OPTIM_BETAS="${OPTIM_BETAS:-(0.95, 0.999)}"
OPTIM_EPS="${OPTIM_EPS:-1e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
LOG_FREQ="${LOG_FREQ:-1000}"

# ──────────── LR Scheduler ────────────
SCHEDULER_NAME="${SCHEDULER_NAME:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"

# ═══════════════════════════════════════════════
printf '\n========== LeRobot Diffusion 训练启动 =========='
printf '\n数据集路径: %s' "$DATASET_ROOT"
printf '\n数据集 repo_id: %s' "$DATASET_REPO_ID"
printf '\n输出目录: %s' "$OUTPUT_DIR"
printf '\njob_name: %s' "$JOB_NAME"
printf '\n训练步数: %s' "$STEPS"
printf '\nbatch size: %s' "$BATCH_SIZE"
printf '\nnum workers: %s' "$NUM_WORKERS"
printf '\nsave freq: %s' "$SAVE_FREQ"
printf '\neval freq: %s' "$EVAL_FREQ"
printf '\ndevice: %s' "$POLICY_DEVICE"
printf '\nwandb: %s' "$WANDB_ENABLE"
# ---------- 新增: 优化器 / 调度器 ----------
printf '\n--- Optimizer ---'
printf '\nLR: %s' "$LR"
printf '\nbetas: %s' "$OPTIM_BETAS"
printf '\neps: %s' "$OPTIM_EPS"
printf '\nweight_decay: %s' "$WEIGHT_DECAY"
printf '\ngrad_clip_norm: %s' "$GRAD_CLIP_NORM"
printf '\n--- Scheduler ---'
printf '\nscheduler: %s' "$SCHEDULER_NAME"
printf '\nwarmup_steps: %s' "$WARMUP_STEPS"
printf '\n=============================================\n\n'

lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=diffusion \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.crop_shape='[480,640]' \
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
  \
  # ── Optimizer ──
  --policy.optimizer_lr="${LR}" \
  --policy.optimizer_betas="${OPTIM_BETAS}" \
  --policy.optimizer_eps="${OPTIM_EPS}" \
  --policy.optimizer_weight_decay="${WEIGHT_DECAY}" \
  \
  # ── Scheduler ──
  --policy.scheduler_name="${SCHEDULER_NAME}" \
  --policy.scheduler_warmup_steps="${WARMUP_STEPS}"
