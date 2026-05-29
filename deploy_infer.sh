#!/usr/bin/env bash
set -euo pipefail

# Pure inference for Piper real robot with a trained diffusion policy.
# This script does NOT create a dataset or record episodes.
#
# Usage:
#   bash deploy_infer.sh
#
# Optional environment variables:
#   POLICY_PATH=/home/bbncf305/lerobot513/DP/lerobot_piper3/outputs/train/diffusion_piper1/checkpoints/last/pretrained_model
#   CAM_AZURE=/dev/video6
#   CAM_REALSENSE=/dev/video4
#   ROBOT_ID=black
#   FPS=30
#   MAX_STEPS=0   # 0 means run forever until Ctrl+C
#   USE_AMP=false
#   POLICY_DEVICE=cuda
#   DEBUG=true
#   ACTION_SCALE=1.0         # 只缩放 6 个机械臂关节动作幅度，夹爪保持原策略输出
#   CONTROL_HZ=15            # 控制循环频率；与相机 FPS 区分开，决定每秒推理和发送动作的次数
#   LPF_ALPHA=1.0            # 动作低通滤波系数；1.0 表示不滤波，越小越平滑但响应越慢，建议先试 0.2~0.5
#   SEND_REPEAT=1            # 每次推理得到的同一个动作重复发送次数；1 表示只发送一次，可用于测试通信/执行稳定性
#   STUCK_DETECT_WINDOW=0    # 卡住检测窗口步数；0 表示关闭，>0 时若连续多步关节状态变化很小则触发脱困信号
#   STUCK_ESCAPE_STEPS=3     # 触发卡住后连续输出脱困动作的步数；用于短暂打破僵局，然后回到策略控制
#   STUCK_ESCAPE_SCALE=0.03  # 脱困动作幅度；会沿着策略想去但机械臂没动起来的方向额外推进，建议从小值开始
#   STUCK_COOLDOWN_STEPS=20  # 每次脱困后的冷却步数；避免在同一位置连续频繁触发脱困

POLICY_PATH="${POLICY_PATH:-/home/bbncf305/lerobot513/DP/lerobot_piper3/outputs/train/diffusion_piper2/checkpoints/060000/pretrained_model}"
CAM_AZURE="${CAM_AZURE:-/dev/video6}"
CAM_REALSENSE="${CAM_REALSENSE:-/dev/video4}"
ROBOT_ID="${ROBOT_ID:-black}"
FPS="${FPS:-15}"
MAX_STEPS="${MAX_STEPS:-0}"
USE_AMP="${USE_AMP:-false}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
DEBUG="${DEBUG:-false}"
ACTION_SCALE="${ACTION_SCALE:-1}"
CONTROL_HZ="${CONTROL_HZ:-15}"
LPF_ALPHA="${LPF_ALPHA:-0.8}"
SEND_REPEAT="${SEND_REPEAT:-1}"
STUCK_DETECT_WINDOW="${STUCK_DETECT_WINDOW:-5}"
STUCK_ESCAPE_STEPS="${STUCK_ESCAPE_STEPS:-8}"
STUCK_ESCAPE_SCALE="${STUCK_ESCAPE_SCALE:-0.3}"
STUCK_COOLDOWN_STEPS="${STUCK_COOLDOWN_STEPS:-1}"

printf '\n========== LeRobot 纯推理启动 =========='
printf '\npolicy 路径: %s' "$POLICY_PATH"
printf '\n相机: azure_kinect=%s, realsense=%s' "$CAM_AZURE" "$CAM_REALSENSE"
printf '\nrobot id: %s' "$ROBOT_ID"
printf '\nfps: %s' "$FPS"
printf '\nmax steps: %s (0 表示无限循环)' "$MAX_STEPS"
printf '\nuse_amp: %s' "$USE_AMP"
printf '\npolicy_device: %s' "$POLICY_DEVICE"
printf '\ndebug: %s' "$DEBUG"
printf '\naction_scale: %s' "$ACTION_SCALE"
printf '\ncontrol_hz: %s' "$CONTROL_HZ"
printf '\nlpf_alpha: %s' "$LPF_ALPHA"
printf '\nsend_repeat: %s' "$SEND_REPEAT"
printf '\nstuck_detect_window: %s' "$STUCK_DETECT_WINDOW"
printf '\nstuck_escape_steps: %s' "$STUCK_ESCAPE_STEPS"
printf '\nstuck_escape_scale: %s' "$STUCK_ESCAPE_SCALE"
printf '\nstuck_cooldown_steps: %s' "$STUCK_COOLDOWN_STEPS"
printf '\n======================================\n\n'

python - "$POLICY_PATH" "$CAM_AZURE" "$CAM_REALSENSE" "$ROBOT_ID" "$FPS" "$MAX_STEPS" "$USE_AMP" "$POLICY_DEVICE" "$DEBUG" "$ACTION_SCALE" "$CONTROL_HZ" "$LPF_ALPHA" "$SEND_REPEAT" "$STUCK_DETECT_WINDOW" "$STUCK_ESCAPE_STEPS" "$STUCK_ESCAPE_SCALE" "$STUCK_COOLDOWN_STEPS" <<'PY'
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.piper.piper import PiperMotorsBusConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots import make_robot_from_config
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import get_safe_torch_device

register_third_party_plugins()

(
    policy_path,
    cam_azure,
    cam_realsense,
    robot_id,
    fps,
    max_steps,
    use_amp,
    policy_device,
    debug,
    action_scale,
    control_hz,
    lpf_alpha,
    send_repeat,
    stuck_detect_window,
    stuck_escape_steps,
    stuck_escape_scale,
    stuck_cooldown_steps,
) = sys.argv[1:18]
policy_path = Path(policy_path)
fps = int(fps)
max_steps = int(max_steps)
use_amp = use_amp.lower() == "true"
debug = debug.lower() == "true"
action_scale = float(action_scale)
control_hz = float(control_hz)
lpf_alpha = float(lpf_alpha)
send_repeat = int(send_repeat)
stuck_detect_window = int(stuck_detect_window)
stuck_escape_steps = int(stuck_escape_steps)
stuck_escape_scale = float(stuck_escape_scale)
stuck_cooldown_steps = int(stuck_cooldown_steps)

if fps <= 0:
    raise ValueError(f"FPS must be > 0, got {fps}")
if control_hz <= 0:
    raise ValueError(f"CONTROL_HZ must be > 0, got {control_hz}")
if not 0.0 < lpf_alpha <= 1.0:
    raise ValueError(f"LPF_ALPHA must be in (0, 1], got {lpf_alpha}")
if send_repeat < 1:
    raise ValueError(f"SEND_REPEAT must be >= 1, got {send_repeat}")
if stuck_detect_window < 0:
    raise ValueError(f"STUCK_DETECT_WINDOW must be >= 0, got {stuck_detect_window}")
if stuck_escape_steps < 0:
    raise ValueError(f"STUCK_ESCAPE_STEPS must be >= 0, got {stuck_escape_steps}")
if stuck_escape_scale < 0:
    raise ValueError(f"STUCK_ESCAPE_SCALE must be >= 0, got {stuck_escape_scale}")
if stuck_cooldown_steps < 0:
    raise ValueError(f"STUCK_COOLDOWN_STEPS must be >= 0, got {stuck_cooldown_steps}")


def _print_debug(prefix, x):
    if not debug:
        return
    if isinstance(x, dict):
        keys = list(x.keys())
        print(f"[DEBUG] {prefix}: dict keys={keys}")
        for k, v in x.items():
            if torch.is_tensor(v):
                print(f"[DEBUG]   {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
            else:
                print(f"[DEBUG]   {k}: type={type(v).__name__}")
    elif torch.is_tensor(x):
        print(f"[DEBUG] {prefix}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
    else:
        print(f"[DEBUG] {prefix}: type={type(x).__name__}")

robot_cfg = PiperFollowerConfig(
    id=robot_id,
    motors=PiperMotorsBusConfig(
        can_name="can0",
        motors={
            "joint_1": (1, "agilex_piper"),
            "joint_2": (2, "agilex_piper"),
            "joint_3": (3, "agilex_piper"),
            "joint_4": (4, "agilex_piper"),
            "joint_5": (5, "agilex_piper"),
            "joint_6": (6, "agilex_piper"),
            "gripper": (7, "agilex_piper"),
        },
    ),
    cameras={
        "azure_kinect": OpenCVCameraConfig(index_or_path=cam_azure, width=1280, height=720, fps=30),
        "realsense": OpenCVCameraConfig(index_or_path=cam_realsense, width=640, height=480, fps=30),
    },
)

robot = make_robot_from_config(robot_cfg)

policy_cls = get_policy_class("diffusion")
policy = policy_cls.from_pretrained(policy_path)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=str(policy_path),
    preprocessor_overrides={"device_processor": {"device": policy_device}},
)

device = get_safe_torch_device(policy_device)
policy.to(device)
policy.eval()

# Motor order for Piper (matches training data)
motor_order = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
motor_names = [f"{m}.pos" for m in motor_order]

print("[INFO] policy loaded")
print(f"[INFO] policy class={policy.__class__.__name__}")
print(f"[INFO] policy device={device}")
print(f"[INFO] policy n_obs_steps={policy.config.n_obs_steps}")
print(f"[INFO] policy horizon={policy.config.horizon}")
print(f"[INFO] policy n_action_steps={policy.config.n_action_steps}")
print(f"[INFO] input features={list(policy.config.input_features.keys())}")
print(f"[INFO] output features={list(policy.config.output_features.keys())}")

stop = False


def _handle_sigint(signum, frame):
    global stop
    stop = True


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)

robot.connect()
print("\n[INFO] Connected. Starting pure inference loop for follower arm only. Press Ctrl+C to stop.\n")

step = 0
last_filtered_action = None
state_history = deque(maxlen=stuck_detect_window if stuck_detect_window > 0 else 1)
stuck_state_epsilon = 1e-3
stuck_escape_remaining = 0
stuck_cooldown_remaining = 0
stuck_escape_direction = None
control_dt = 1.0 / control_hz
try:
    while not stop:
        loop_t = time.perf_counter()
        raw_obs = robot.get_observation()
        _print_debug("raw_obs", raw_obs)

        # Step 1: Convert raw observation to canonical LeRobot format
        #   raw_obs: {'joint_1.pos': 0.1, 'azure_kinect': ndarray(H,W,C)}
        #   canonical: {'observation.state': np.array([...]), 'observation.images.azure_kinect': ndarray(H,W,C)}
        canonical_obs = {}
        current_state = np.array(
            [raw_obs[name] for name in motor_names], dtype=np.float32
        )
        canonical_obs["observation.state"] = current_state
        for cam_key in robot.cameras:
            canonical_obs[f"observation.images.{cam_key}"] = raw_obs[cam_key]
        _print_debug("canonical_obs", canonical_obs)

        # Step 2: Use the standard LeRobot inference flow:
        #   prepare_observation_for_inference: numpy->tensor, normalize images, add batch dim
        #   preprocessor (via __call__): runs the full pipeline (batch, device, normalize)
        obs = prepare_observation_for_inference(
            canonical_obs, device, task=None, robot_type=None
        )
        _print_debug("after_prepare", obs)

        obs = preprocessor(obs)
        _print_debug("preprocessed_obs", obs)

        # Step 3: Inference
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            action = policy.select_action(obs)

        _print_debug("policy_action", action)

        # Step 4: Postprocess action (unnormalize, move to cpu)
        action = postprocessor(action)
        _print_debug("postprocessed_action", action)

        # Step 5: Convert policy action tensor back to robot action dict
        action_tensor = action.squeeze(0).cpu().float()

        # 只对机械臂 6 个关节做缩放，gripper 保持原策略输出，避免夹爪值被放大后溢出。
        if action_scale != 1.0:
            action_tensor[:6] = action_tensor[:6] * action_scale

        # 对完整动作做一阶低通滤波，降低相邻控制步之间的高频抖动。
        if lpf_alpha < 1.0:
            if last_filtered_action is None:
                filtered_action = action_tensor.clone()
            else:
                filtered_action = lpf_alpha * action_tensor + (1.0 - lpf_alpha) * last_filtered_action
            last_filtered_action = filtered_action.clone()
            action_tensor = filtered_action

        if stuck_cooldown_remaining > 0:
            stuck_cooldown_remaining -= 1

        if stuck_escape_remaining > 0 and stuck_escape_direction is not None:
            # 脱困信号：在策略动作基础上，沿“策略想去但机械臂没动起来”的方向额外推进一点。
            # 只作用于 6 个机械臂关节，不改 gripper，避免夹爪被脱困逻辑误触发。
            action_tensor[:6] = action_tensor[:6] + stuck_escape_scale * stuck_escape_direction
            stuck_escape_remaining -= 1
            print(f"[ESCAPE] 正在输出脱困动作: remaining={stuck_escape_remaining}, scale={stuck_escape_scale}")
            if last_filtered_action is not None:
                last_filtered_action = action_tensor.clone()

        robot_action = {f"{motor}.pos": float(action_tensor[i]) for i, motor in enumerate(motor_order)}
        _print_debug("robot_action", robot_action)

        sent_action = None
        for repeat_idx in range(send_repeat):
            sent_action = robot.send_action(robot_action)
            if debug and send_repeat > 1:
                _print_debug(f"sent_action[{repeat_idx + 1}/{send_repeat}]", sent_action)
        if not (debug and send_repeat > 1):
            _print_debug("sent_action", sent_action)

        if stuck_detect_window > 0:
            state_history.append(current_state.copy())
            if len(state_history) == stuck_detect_window:
                state_delta = float(np.max(np.abs(state_history[-1] - state_history[0])))
                desired_delta = action_tensor.detach().cpu().numpy()[:6] - current_state[:6]
                action_delta = float(np.max(np.abs(desired_delta)))
                can_escape = (
                    stuck_escape_steps > 0
                    and stuck_escape_scale > 0.0
                    and stuck_escape_remaining == 0
                    and stuck_cooldown_remaining == 0
                )
                if state_delta < stuck_state_epsilon and action_delta > stuck_state_epsilon and can_escape:
                    norm = float(np.linalg.norm(desired_delta))
                    if norm > 1e-9:
                        stuck_escape_direction = torch.from_numpy(desired_delta / norm).to(action_tensor)
                        stuck_escape_remaining = stuck_escape_steps
                        stuck_cooldown_remaining = stuck_cooldown_steps
                        state_history.clear()
                        print(
                            f"[ESCAPE] 检测到卡住，开始输出脱困信号: window={stuck_detect_window}, "
                            f"state_delta={state_delta:.6f}, action_delta={action_delta:.6f}, "
                            f"escape_steps={stuck_escape_steps}, escape_scale={stuck_escape_scale}"
                        )
                elif state_delta < stuck_state_epsilon and action_delta > stuck_state_epsilon:
                    print(
                        f"[WARN] 可能卡住但暂不触发脱困: state_delta={state_delta:.6f}, "
                        f"action_delta={action_delta:.6f}, escape_remaining={stuck_escape_remaining}, "
                        f"cooldown_remaining={stuck_cooldown_remaining}"
                    )

        step += 1
        if step % 10 == 0:
            print(f"[INFO] step={step}")
        if max_steps > 0 and step >= max_steps:
            print(f"[INFO] Reached MAX_STEPS={max_steps}, stopping.")
            break

        time.sleep(max(control_dt - (time.perf_counter() - loop_t), 0.0))
finally:
    try:
        robot.disconnect()
    except Exception:
        pass
    print("[INFO] Clean exit.")
PY
