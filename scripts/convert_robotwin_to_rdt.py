import os
import glob
import h5py
import json
import random
import numpy as np
import cv2
import webdataset as wds
from tqdm import tqdm

# ================= 配置区域 =================
DATASET_ROOT = "/home/jiangjiahao/data/aloha_randomized_subset"
OUTPUT_DIR = "/home/jiangjiahao/data/rdt2_robotwin_shards_test"

# [新] 生成的全局指令文件路径
OUTPUT_INSTRUCTION_JSON = os.path.join(OUTPUT_DIR, "robotwin_instruction.json")

# 其他配置
CHUNK_SIZE = 24
RESIZE_H, RESIZE_W = 384, 384
CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
# ===========================================

def load_instructions(json_path):
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if "instructions" in data: return data["instructions"]
        elif isinstance(data, list): return data
        return None
    except: return None

def decode_image(byte_data):
    # (保持之前的解码逻辑不变)
    valid_bytes = byte_data.rstrip(b'\0')
    data = np.frombuffer(valid_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        data_raw = np.frombuffer(byte_data, dtype=np.uint8)
        img = cv2.imdecode(data_raw, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("Decode failed")
    return img

# 全局指令字典 {unique_key: instruction_text}
GLOBAL_INSTRUCTION_DICT = {}

def process_episode(file_path, instruction_list):
    samples = []
    
    # 1. 确定指令文本
    if instruction_list and len(instruction_list) > 0:
        text = random.choice(instruction_list)
    # else:
    #     text = "do task"
    
    # 2. 生成唯一 Key (TaskName + EpisodeName)
    # 路径结构: .../TaskName/EpisodeDir/file.hdf5
    episode_dir_name = os.path.basename(os.path.dirname(file_path))
    task_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    
    # 这是一个全局唯一的 Key
    instr_key = f"{task_name}_{episode_dir_name}"
    
    # 3. 存入全局字典
    GLOBAL_INSTRUCTION_DICT[instr_key] = text
    
    with h5py.File(file_path, 'r') as root:
        try:
            actions = root["/action"][()]
            qpos = root["/observations/qpos"][()]
        except KeyError: return []
            
        episode_len = len(actions)
        if episode_len < CHUNK_SIZE: return []

        # 读取图像
        image_data = {}
        for cam in CAMERA_NAMES:
            key = f"/observations/images/{cam}"
            if key in root: image_data[cam] = root[key][()]
            # else: return []

        for t in range(episode_len - CHUNK_SIZE):
            # 图像处理
            img_list = []
            for cam in CAMERA_NAMES:
                raw_bytes = image_data[cam][t]
                img = decode_image(raw_bytes)
                if img.shape[0] != RESIZE_H or img.shape[1] != RESIZE_W:
                    img = cv2.resize(img, (RESIZE_W, RESIZE_H))
                img_list.append(img)
            concat_img = np.concatenate(img_list, axis=1)

            # 动作 & 状态
            action_chunk = actions[t : t + CHUNK_SIZE].astype(np.float32)
            state_frame = qpos[t].astype(np.float32)

            key = f"{instr_key}_{t:06d}"
            
            sample = {
                "__key__": key,
                "image.jpg": concat_img,
                "action.npy": action_chunk,
                "state.npy": state_frame,
                # 占位符 token (RDT-FM 不用)
                "action_token.npy": np.zeros((1,), dtype=np.int16),
                
                "meta.json": {
                    "frame_id": t,
                    "episode_id": episode_dir_name,
                    # [关键修改] 这里存 Key，而不是 Text！
                    "sub_task_instruction_key": instr_key 
                }
            }
            samples.append(sample)
    return samples

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    total_samples_written = 0  # [新增] 统计写入的 sample 数
    total_episodes_used = 0    # [可选] 统计实际处理并产生样本的 episode 数

    print(f"Scanning {DATASET_ROOT}...")
    task_dirs = [d for d in glob.glob(os.path.join(DATASET_ROOT, "*")) if os.path.isdir(d)]
    
    pattern = os.path.join(OUTPUT_DIR, "shard-%06d.tar")
    with wds.ShardWriter(pattern, maxcount=2000, maxsize=5e8) as sink:
        for task_dir in tqdm(task_dirs, desc="Tasks"):
            episode_dirs = [d for d in glob.glob(os.path.join(task_dir, "*")) if os.path.isdir(d)]
            for ep_dir in episode_dirs:
                hdf5_files = glob.glob(os.path.join(ep_dir, "*.hdf5"))
                if not hdf5_files: continue
                
                json_path = os.path.join(ep_dir, "instructions.json")
                instrs = load_instructions(json_path)
                
                try:
                    samples = process_episode(hdf5_files[0], instrs)
                    if samples: # 确保有数据才计数
                        for s in samples: 
                            sink.write(s)
                            total_samples_written += 1  # <--- 加上这行
                        total_episodes_used += 1        # <--- 加上这行
                except Exception as e:
                    print(f"Error: {e}")

    # [最后] 保存生成的全局指令文件
    print(f"Saving combined instruction file to: {OUTPUT_INSTRUCTION_JSON}")
    with open(OUTPUT_INSTRUCTION_JSON, 'w') as f:
        json.dump(GLOBAL_INSTRUCTION_DICT, f, indent=2)
    # [新增] 打印统计信息
    print(f"Total samples written: {total_samples_written}")
    print(f"Total episodes used: {total_episodes_used}")
    print(f"Total unique instruction keys: {len(GLOBAL_INSTRUCTION_DICT)}")
    
    print("Done!")

if __name__ == "__main__":
    main()