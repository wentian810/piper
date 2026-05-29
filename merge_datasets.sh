#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# 合并多个 LeRobot 数据集为一个多任务数据集
# 用法:
#   bash merge_datasets.sh
#
# 每个源数据集的 task 会自动分配不同的 task_index，
# 合并后训练时模型通过 task_index 区分任务。
# ============================================================

# ── 配置 ──
DATASETS_DIR="${DATASETS_DIR:-/data1/jxyu26/lerobot_piper3/datasets}"
SOURCE_NAMES="${SOURCE_NAMES:-piper1 piper2 piper3 piper4}"
AGGR_NAME="${AGGR_NAME:-piper_all}"

# ═══════════════════════════════════════════════
printf '\n========== 合并数据集 =========='
printf '\n数据集目录: %s' "$DATASETS_DIR"
printf '\n源数据集: %s' "$SOURCE_NAMES"
printf '\n合并输出: %s' "$AGGR_NAME"
printf '\n================================\n\n'

python -c "
import sys
from pathlib import Path

datasets_dir = Path('${DATASETS_DIR}')
source_names = '${SOURCE_NAMES}'.split()
aggr_name = '${AGGR_NAME}'

repo_ids = [f'local/{name}' for name in source_names]
roots = [datasets_dir / name for name in source_names]
aggr_root = datasets_dir / aggr_name

print(f'源数据集路径:')
for root in roots:
    if not root.exists():
        print(f'  错误: {root} 不存在！')
        sys.exit(1)
    print(f'  {root}  OK')

print(f'合并目标路径: {aggr_root}')

if aggr_root.exists():
    print(f'错误: 目标路径 {aggr_root} 已存在，请先手动删除')
    sys.exit(1)

from lerobot.datasets.aggregate import aggregate_datasets
aggregate_datasets(
    repo_ids=repo_ids,
    aggr_repo_id=f'local/{aggr_name}',
    roots=roots,
    aggr_root=aggr_root,
)

print(f'\n合并完成！')
print(f'数据集路径: {aggr_root}')
print(f'repo_id:     local/{aggr_name}')
"
