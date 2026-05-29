#!/usr/bin/env python
"""
Piper + Diffusion Policy 实时推理脚本

用法:
    # 有相机 (video0=azure_kinect, video6=realsense)
    python inference_piper.py \
        --checkpoint /home/bbncf305/lerobot513/DP/outputs/train/task2_new_diffusion/checkpoints/180000/pretrained_model \
        --camera-devices azure_kinect=/dev/video0,realsense=/dev/video6

    # 无相机（仅用 state）
    python inference_piper.py \
        --checkpoint /path/to/pretrained_model \
        --no-camera
"""

import argparse
import time

import numpy as np
import torch
import torchvision.transforms as T

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.robots import make_robot_from_config, piper_follower  # noqa: F401
from lerobot.robots.config import RobotConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device


def load_policy(checkpoint_path: str, device: str = "cuda") -> DiffusionPolicy:
    """加载训练好的 Diffusion Policy checkpoint"""
    print(f"Loading policy from {checkpoint_path} ...")
    policy = DiffusionPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()
    print(f"Policy loaded! Device: {next(policy.parameters()).device}")
    return policy


def parse_camera_devices(s: str | None) -> dict[str, str]:
    """解析摄像头设备映射: --camera-devices azure_kinect=/dev/video0,realsense=/dev/video6"""
    if not s:
        return {}
    mapping = {}
    for pair in s.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            mapping[k.strip()] = v.strip()
    return mapping


def main():
    register_third_party_plugins()

    parser = argparse.ArgumentParser(description="Piper + Diffusion Policy 实时推理")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint 路径 (pretrained_model 目录)")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备 (cuda / cpu)")
    parser.add_argument("--fps", type=int, default=30, help="推理帧率")
    parser.add_argument("--no-camera", action="store_true", help="无相机模式（仅使用关节状态）")
    parser.add_argument("--camera-devices", type=str, default=None,
                        help="摄像头设备映射，如 'azure_kinect=/dev/video0,realsense=/dev/video6'")
    args = parser.parse_args()

    device = get_safe_torch_device(args.device)
    camera_devices = parse_camera_devices(args.camera_devices)

    print(f"Using device: {device}")
    if args.no_camera:
        print("No-camera mode: only using joint states")
    if camera_devices:
        print(f"Camera devices: {camera_devices}")

    # ====== 1. 加载 policy ======
    policy = load_policy(args.checkpoint, device)

    # 检查 policy 需要的视觉输入
    required_cameras = [k for k in policy.config.input_features if k.startswith("observation.images.")]
    print(f"Policy requires cameras: {required_cameras}")

    if not args.no_camera and len(required_cameras) > 0 and len(camera_devices) == 0:
        print("ERROR: Policy requires cameras but no --camera-devices specified!")
        print("Usage example: --camera-devices azure_kinect=/dev/video0,realsense=/dev/video6")
        return

    # ====== 2. 连接 piper 机器人 ======
    print("Connecting to Piper robot...")

    # 构建带摄像头配置的 PiperFollowerConfig
    from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig

    robot_config = PiperFollowerConfig()
    robot_config.motors.can_name = "can0"

    if not args.no_camera:
        for cam_name, device_path in camera_devices.items():
            if cam_name == "azure_kinect":
                robot_config.cameras[cam_name] = OpenCVCameraConfig(
                    index_or_path=device_path,
                    fps=args.fps,
                    width=1280,
                    height=720,
                    fourcc="YUYV",
                )
                print(f"  Configured '{cam_name}' -> {device_path} (1280x720)")
            elif cam_name == "realsense":
                robot_config.cameras[cam_name] = OpenCVCameraConfig(
                    index_or_path=device_path,
                    fps=args.fps,
                    width=640,
                    height=480,
                    fourcc="YUYV",
                )
                print(f"  Configured '{cam_name}' -> {device_path} (640x480)")
            else:
                robot_config.cameras[cam_name] = OpenCVCameraConfig(
                    index_or_path=device_path,
                    fps=args.fps,
                )
                print(f"  Configured '{cam_name}' -> {device_path}")

    robot = make_robot_from_config(robot_config)
    robot.connect()
    print("Piper connected and calibrated!")

    # ====== 3. 推理循环 ======
    print("\n=== Starting inference loop ===")
    print("Press Ctrl+C to stop.\n")

    motor_order = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]

    policy.reset()
    step = 0
    try:
        with torch.inference_mode():
            while True:
                loop_start = time.perf_counter()

                # 3a. 获取机器人观测（关节状态 + 图像）
                robot_obs = robot.get_observation()

                # 3b. 构造 policy 输入（关节状态）
                state = np.array([robot_obs[f"{m}.pos"] for m in motor_order], dtype=np.float32)
                batch = {
                    "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),  # [1, 7]
                }

                # 3c. 添加相机图像（如果需要）
                if not args.no_camera and robot.has_camera:
                    for train_cam_key in required_cameras:
                        cam_short_name = train_cam_key.replace("observation.images.", "")

                        if cam_short_name in robot_obs:
                            img = robot_obs[cam_short_name]
                            if isinstance(img, np.ndarray) and img.size > 0:
                                # (H, W, C) -> (1, C, H, W)
                                img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
                                # Center crop to 440x560 (same as training)
                                if img_tensor.shape[-2:] != (440, 560):
                                    crop = T.CenterCrop((440, 560))
                                    img_tensor = crop(img_tensor)
                                batch[train_cam_key] = img_tensor
                            else:
                                print(f"Warning: camera {cam_short_name} returned empty data")
                        else:
                            print(f"Warning: camera '{cam_short_name}' not found. Available: {list(robot_obs.keys())}")

                # 3d. Policy 推理 -> 选动作
                action = policy.select_action(batch)
                action_np = action.cpu().numpy().flatten()  # [7]

                # 3e. 发送动作到机器人
                action_dict = {f"{m}.pos": float(action_np[i]) for i, m in enumerate(motor_order)}
                robot.send_action(action_dict)

                # 3f. 打印信息
                if step % 10 == 0:
                    state_str = ", ".join([f"{action_np[i]:.3f}" for i in range(7)])
                    fps = 1.0 / (time.perf_counter() - loop_start + 1e-6)
                    print(f"[step {step:>4d}] action: [{state_str}]    fps={fps:.1f}")

                step += 1

                # 3g. 保持目标帧率
                dt = time.perf_counter() - loop_start
                precise_sleep(max(1.0 / args.fps - dt, 0.0))

    except KeyboardInterrupt:
        print("\n\nStopping inference...")
    finally:
        robot.disconnect()
        print("Robot disconnected. Done.")


if __name__ == "__main__":
    main()
