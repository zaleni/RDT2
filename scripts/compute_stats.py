import torch
import numpy as np
import webdataset as wds
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.normalizer.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

# ================= 配置 =================
# 请替换为您的实际路径
DATASET_PATH = "/home/jiangjiahao/data/rdt2_robotwin/rdt2_robotwin_shards/shard-{000000..000247}.tar"
OUTPUT_PT_PATH = "/home/jiangjiahao/data/rdt2_robotwin/robotwin_joint_14d_normalizer.pt"
ACTION_DIM = 14
MAX_SAMPLES = None  # 设置为 None 则使用全部样本
# 防止除以0
EPS = 1e-6
# =======================================

def main():
    # [关键点 1] 使用 numpy 解码器，确保正确读取 .npy 文件
    dataset = wds.WebDataset(DATASET_PATH).decode().to_tuple("action.npy")

    all_actions = []
    print("Collecting samples for normalization...")

    for i, (action,) in tqdm(enumerate(dataset)):
        if MAX_SAMPLES is not None and i >= MAX_SAMPLES:
            break

        # 确保转为 numpy 数组
        action = np.asarray(action)
        
        # [关键点 2] 增加形状检查，防止脏数据导致后续报错
        if action.shape[-1] != ACTION_DIM:
            print(f"Warning: Skipping sample {i} with unexpected shape {action.shape}")
            continue

        all_actions.append(action)

    if len(all_actions) == 0:
        print("No data found!")
        return

    # 转换为 Tensor 并展平: (N * T, D)
    all_actions = np.asarray(all_actions, dtype=np.float32)
    x = torch.from_numpy(all_actions).float().view(-1, ACTION_DIM)

    print(f"Calculating stats on shape: {x.shape}")

    # [关键点 3] 使用 Min-Max 归一化到 [-1, 1] (符合 RDT2 习惯)
    # 1. 计算统计量
    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0) # 仅用于记录，不用于计算 scale

    # 2. 计算 Scale 和 Offset
    range_val = x_max - x_min
    range_val[range_val < EPS] = 1.0  # 防止除以0
    
    scale = 2.0 / range_val
    offset = -1.0 - x_min * scale

    # 3. 创建参数对象
    input_stats = {
        "min": x_min,
        "max": x_max,
        "mean": x_mean,
        "std": x_std,
    }
    
    action_norm = SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=input_stats,
    )

    # 4. 封装并保存
    normalizer = LinearNormalizer()
    normalizer["action"] = action_norm 

    # 打印检查
    print("Mean:", normalizer.params_dict["action"]["input_stats"]["mean"][:4])
    print("Scale:", normalizer.params_dict["action"]["scale"][:4])

    # [关键点 4] 正确保存
    normalizer.save(OUTPUT_PT_PATH)
    print(f"Saved normalizer to {OUTPUT_PT_PATH}")

if __name__ == "__main__":
    main()