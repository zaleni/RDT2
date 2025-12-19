import torch
import numpy as np # 需要导入 numpy
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from vqvae.models.multivqvae import MultiVQVAE
from models.normalizer import LinearNormalizer
from utils import batch_predict_action

# 假设使用 gpu 0
device = "cuda:3"

# 1. 加载 Processor 和 VLM 模型
# 确保路径存在
processor = AutoProcessor.from_pretrained("/home/jiangjiahao/data/model/Qwen2.5-VL-7B-Instruct")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/home/jiangjiahao/data/model/RDT2-VQ",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device
).eval()

# 2. 加载 VAE
# MultiVQVAE 继承自 HuggingFace Mixin，可以使用 from_pretrained
vae = MultiVQVAE.from_pretrained("/home/jiangjiahao/data/model/RVQActionTokenizer").eval()
vae = vae.to(device=device, dtype=torch.float32)

valid_action_id_length = (
    vae.pos_id_len + vae.rot_id_len + vae.grip_id_len
)

# 3. [关键修改] 加载 Normalizer
# 源代码中 LinearNormalizer 定义的是 load() 方法，而不是 from_pretrained()
# 请确保 "umi_normalizer_wo_downsample_indentity_rot.pt" 文件在当前目录下，或者填写绝对路径
normalizer = LinearNormalizer.load("umi_normalizer_wo_downsample_indentity_rot.pt") # <--- 修改这里

# 4. [关键修改] 构造真实的输入数据
# 您代码中的 ... 需要替换为真实的 numpy 数组 (T, H, W, C)
# 这里构造一个全零的假数据用于测试跑通
fake_image_np = np.zeros((1, 384, 384, 3), dtype=np.uint8) 
fake_image = torch.from_numpy(fake_image_np).to(device)

result = batch_predict_action(
    model,
    processor,
    vae,
    normalizer,
    examples=[
        {
            "obs": {
                # 这里的 key 需要对应 configs/dataset 里的 camera_names
                "camera0_rgb": fake_image, # <--- 替换 ... 为真实数据
                "camera1_rgb": fake_image, # <--- 替换 ... 为真实数据
            },
            "meta": {
                "dataset_name": "test", # batch_predict_action 内部解包时可能需要这个字段
                "num_camera": 2,
                "num_robot": 1 # 或者 2，取决于您的具体配置
            }
        }
    ],
    valid_action_id_length=valid_action_id_length,
    apply_jpeg_compression=False, # 测试时可以先关掉
    instruction="Pick up the apple."
)

# 获取预测动作
action_chunk = result["action_pred"][0] 
print("预测动作形状:", action_chunk.shape)

# 后处理：Rescale gripper width
# 假设是双臂机器人配置
if action_chunk.shape[-1] >= 20: 
    for robot_idx in range(2):
        # 确保索引不越界
        idx = robot_idx * 10 + 9
        if idx < action_chunk.shape[-1]:
            action_chunk[:, idx] = action_chunk[:, idx] / 0.088 * 0.1

print("代码运行成功！")