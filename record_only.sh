#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash record_only.sh
#
# Optional environment variables:
#   HF_USER=local
#   DATASET_NAME=piper1
#   NUM_EPISODES=50
#   EPISODE_TIME_S=45
#   RESET_TIME_S=30
#   DATASET_ROOT=/home/bbncf305/lerobot513/DP/lerobot_piper3/datasets/piper1
#   sudo ip link set can0 up type can bitrate 1000000
#   sudo ip link set can1 up type can bitrate 1000000

HF_USER="${HF_USER:-local}"
DATASET_NAME="${DATASET_NAME:-piper4}"
NUM_EPISODES="${NUM_EPISODES:-50}"
EPISODE_TIME_S="${EPISODE_TIME_S:-45}"
RESET_TIME_S="${RESET_TIME_S:-30}"
DATASET_ROOT="${DATASET_ROOT:-/home/bbncf305/lerobot513/DP/lerobot_piper3/datasets/${DATASET_NAME}}"

printf '\n========== LeRobot 数据录制启动 =========='
printf '\n本地保存路径: %s' "$DATASET_ROOT"
printf '\n数据集名(repo_id): %s/%s' "$HF_USER" "$DATASET_NAME"
printf '\n每个 episode 录制时长: %s 秒' "$EPISODE_TIME_S"
printf '\n场景恢复时长: %s 秒' "$RESET_TIME_S"
printf '\n总 episode 数: %s' "$NUM_EPISODES"
printf '\n相机: azure_kinect=/dev/video6, realsense=/dev/video4'\nprintf '\nCAN: 主臂=can0, 从臂=can1'\nprintf '\n=====================================\n\n'

lerobot-record \
  --robot.type=piper_follower \
  --robot.id=black \
  --robot.cameras="{
    azure_kinect: {type: opencv, index_or_path: '/dev/video6', width: 1280, height: 720, fps: 30},
    realsense: {type: opencv, index_or_path: '/dev/video4', width: 640, height: 480, fps: 30}
  }" \
  --teleop.type=piper_leader \
  --teleop.id=blue \
  --display_data=true \
  --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.push_to_hub=false \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.single_task="pick up the corn and place it to the box" \
  --play_sounds=false 
