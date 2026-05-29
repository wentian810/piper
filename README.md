# Piper 机械臂 — Diffusion Policy 操作指南

三个脚本覆盖完整工作流：**录制数据 → 训练策略 → 部署推理**。

## 硬件连接

| 设备 | CAN 总线 | 设备路径 |
|---|---|---|
| 从臂（被控） | `can0` | — |
| 主臂（遥操作） | `can1` | — |
| Azure Kinect 相机 | — | `/dev/video6` |
| RealSense 相机 | — | `/dev/video4` |

使用前确保 CAN 接口已启用：

```bash
ip link show
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
摄像头需要使用 lerobot-find-cameras 来看序号是否匹配
```

---

## 1. 录制数据 — `record_only.sh`

通过主臂遥操作控制从臂，录制演示数据。

```bash
bash record_only.sh
```

### 可配置环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `HF_USER` | `local` | 数据集所属用户/组织 |
| `DATASET_NAME` | `piper4` | 数据集名称 |
| `NUM_EPISODES` | `50` | 录制 episode 数量 |
| `EPISODE_TIME_S` | `45` | 每个 episode 录制时长（秒） |
| `RESET_TIME_S` | `30` | 场景恢复等待时长（秒） |
| `DATASET_ROOT` | `/home/.../datasets/${DATASET_NAME}` | 数据集保存路径 |

### 示例

```bash
DATASET_NAME=my_task NUM_EPISODES=100 EPISODE_TIME_S=60 bash record_only.sh
```

录制时拖动机器人主臂完成动作，数据自动保存到 `DATASET_ROOT` 指定路径。

---

## 2. 训练策略 — `train_diffusion.sh`

使用录制的数据集训练 Diffusion Policy。

```bash
bash train_diffusion.sh
```

### 可配置环境变量

**数据集 / 输出：**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DATASET_ROOT` | `/home/.../datasets/piper3` | 数据集路径 |
| `DATASET_REPO_ID` | `local/piper3` | 数据集标识 |
| `OUTPUT_DIR` | `outputs/train/diffusion_piper3` | checkpoint 输出目录 |
| `JOB_NAME` | `diffusion_piper3` | 训练任务名称 |

**训练参数：**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `STEPS` | `100000` | 训练总步数 |
| `BATCH_SIZE` | `16` | 批次大小 |
| `LR` | `1e-5` | 学习率 |
| `WARMUP_STEPS` | `500` | 学习率预热步数 |
| `SAVE_FREQ` | `10000` | checkpoint 保存频率 |
| `EVAL_FREQ` | `1000` | 评估频率 |
| `POLICY_DEVICE` | `cuda` | 训练设备 |
| `WANDB_ENABLE` | `false` | 是否启用 wandb 日志 |

**策略结构（已硬编码在脚本中）：**

| 参数 | 值 | 说明 |
|---|---|---|
| `n_obs_steps` | `16` | 观察历史步数（~1s @ 15Hz） |
| `horizon` | `128` | 预测动作总长度（~8.5s） |
| `n_action_steps` | `64` | 每次执行步数（~4.3s），之后重新推理 |
| `drop_n_last_frames` | `49` | 由公式 `horizon - n_action_steps - n_obs_steps + 1` 计算 |

### 示例

```bash
DATASET_ROOT=/path/to/my_dataset \
DATASET_REPO_ID=local/my_dataset \
OUTPUT_DIR=outputs/train/my_task \
STEPS=50000 BATCH_SIZE=32 \
bash train_diffusion.sh
```

训练完成后 checkpoint 保存在 `OUTPUT_DIR/checkpoints/` 下，最终模型使用 `last/pretrained_model`。

---

## 2.5 多任务 Transformer 训练 — `train_diffusion_multi_task.sh`

支持多个任务数据集合并训练，使用 Transformer 架构替代 CNN-UNet，3 卡并行。

### 数据集合并

```bash
# 合并 piper1/2/3/4 为 piper_all，自动分配 task_index
bash merge_datasets.sh
```

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DATASETS_DIR` | `/data1/jxyu26/lerobot_piper3/datasets` | 数据集根目录 |
| `SOURCE_NAMES` | `piper1 piper2 piper3 piper4` | 源数据集名 |
| `AGGR_NAME` | `piper_all` | 合并后名称 |

### 训练

```bash
bash train_diffusion_multi_task.sh
# 单卡: NUM_GPUS=1 bash train_diffusion_multi_task.sh
# 指定 GPU: CUDA_VISIBLE_DEVICES=3,4,5 bash train_diffusion_multi_task.sh
```

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DATASET_ROOT` | `/data1/.../datasets/piper_all` | 合并后数据集路径 |
| `OUTPUT_DIR` | `/data1/.../outputs/train/all` | checkpoint 输出 |
| `NUM_GPUS` | `3` | GPU 数量 |
| `CUDA_VISIBLE_DEVICES` | `3,4,5` | 指定 GPU |
| `STEPS` | `100000` | 训练步数 |
| `BATCH_SIZE` | `16` | 每卡 batch size（有效 = 16×3 = 48） |
| `LR` | `1e-5` | 学习率 |
| `N_OBS_STEPS` | `8` | 观察历史帧数 |
| `HORIZON` | `128` | 预测动作长度 |
| `N_ACTION_STEPS` | `100` | 执行步数 |
| `USE_TRANSFORMER` | `true` | 使用 Transformer（false = CNN-UNet） |
| `N_LAYERS` | `4` | Transformer Decoder 层数 |
| `N_HEADS` | `8` | 注意力头数 |
| `N_EMB` | `256` | 嵌入维度（~4.8M 参数） |
| `NUM_TASKS` | `4` | 任务数 |
| `ACTIVE_TASK_ID` | `0` | 推理时默认任务（训练时自动从数据获取） |

### 关键代码变更

| 文件 | 说明 |
|---|---|
| `configuration_diffusion.py` | 新增 `use_transformer`, `n_layers`, `n_heads`, `n_emb`, `causal_attn`, `num_tasks`, `active_task_id` |
| `modeling_diffusion.py` | 新增 `TransformerForDiffusion` 类 + task embedding；`DiffusionModel` 支持 CNN-UNet / Transformer 切换 |
| `merge_datasets.sh` | 合并多数据集脚本 |
| `train_diffusion_multi_task.sh` | 多任务多卡训练脚本 |

**架构对比：**

| | CNN-UNet | Transformer |
|---|---|---|
| 参数量 | ~278M | ~4.8M |
| 条件注入 | FiLM 通道调制 | 交叉注意力 |
| 显存 | 大 | 小 |
| 多任务 | 不支持 | task embedding |

Transformer 代码基于 [lerobot-joycon](https://github.com/box2ai-robotics/lerobot-joycon) 移植适配。

---

## 3. 部署推理 — `deploy_infer.sh`

加载训练好的策略，在真实从臂上执行推理。

```bash
POLICY_PATH=/path/to/checkpoints/last/pretrained_model bash deploy_infer.sh
```

### 可配置环境变量

**核心参数：**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `POLICY_PATH` | `/home/.../diffusion_piper2/checkpoints/060000/pretrained_model` | 策略 checkpoint 路径 |
| `ROBOT_ID` | `black` | 从臂标识 |
| `POLICY_DEVICE` | `cuda` | 推理设备 |
| `FPS` | `15` | 相机帧率 |
| `CONTROL_HZ` | `15` | 控制循环频率 |
| `MAX_STEPS` | `0`（无限） | 最大推理步数 |

**动作调优：**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ACTION_SCALE` | `1.0` | 机械臂 6 关节动作缩放系数（不影响夹爪） |
| `LPF_ALPHA` | `0.8` | 一阶低通滤波系数（越小越平滑，建议 0.2~0.5） |
| `SEND_REPEAT` | `1` | 每个动作重复发送次数 |

### 示例

```bash
# 基础使用
POLICY_PATH=outputs/train/my_task/checkpoints/last/pretrained_model bash deploy_infer.sh

# 更平滑的动作 + 更保守的动作幅度
POLICY_PATH=outputs/train/my_task/checkpoints/last/pretrained_model \
LPF_ALPHA=0.3 ACTION_SCALE=0.8 bash deploy_infer.sh
```

按 `Ctrl+C` 停止推理，机械臂会自动断开连接。

---

## 完整流程示例

```bash
# Step 1: 录制 100 个 episode
DATASET_NAME=my_task NUM_EPISODES=100 EPISODE_TIME_S=60 bash record_only.sh

# Step 2: 训练 50000 步
DATASET_ROOT=/home/bbncf305/lerobot513/DP/lerobot_piper3/datasets/my_task \
DATASET_REPO_ID=local/my_task \
OUTPUT_DIR=outputs/train/my_task \
STEPS=50000 \
bash train_diffusion.sh

# Step 3: 部署推理
POLICY_PATH=outputs/train/my_task/checkpoints/last/pretrained_model bash deploy_infer.sh
```
