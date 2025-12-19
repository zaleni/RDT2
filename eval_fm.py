import torch
import numpy as np
import yaml
import cv2
from PIL import Image

# 导入 RDT 推理器封装类
# 确保您的 PYTHONPATH 包含了项目根目录，或者 sys.path.append(...)
from models.rdt_inferencer import RDTInferencer

def main():
    # -------------------------------------------------------------------------
    # 1. 配置路径 (根据您的环境修改)
    # -------------------------------------------------------------------------
    # RDT2-FM 模型权重路径 (您训练输出的 output_dir)
    RDT2_FM_PATH = "/home/jiangjiahao/data/ckpt/RDT2/rdt/rdt2-action-expert-robotwin" 
    
    # RDT2-VQ (Qwen2.5-VL) 基础模型路径
    RDT2_VQ_PATH = "/home/jiangjiahao/data/model/RDT2-VQ"
    
    # Normalizer 统计文件 (必须是 14维 的那个)
    NORMALIZER_PATH = "/home/jiangjiahao/data/ckpt/RDT2/rdt/rdt2-action-expert-robotwin/robotwin_joint_14d_normalizer.pt"
    
    # 训练时的配置文件路径
    CONFIG_PATH = "/home/jiangjiahao/research/RDT2/configs/rdt/robotwin_train.yaml"
    
    DEVICE = "cuda:3"
    
    # -------------------------------------------------------------------------
    # 2. 初始化模型
    # -------------------------------------------------------------------------
    print(f"正在加载配置文件: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        model_config = yaml.safe_load(f)

    print("正在初始化 RDT2-FM 推理器...")
    # RDTInferencer 内部会自动处理 Flow Matching (FM) 或 VQ 的加载逻辑
    # 它会读取 config['model'] 来决定网络结构
    model = RDTInferencer(
        config=model_config,
        pretrained_path=RDT2_FM_PATH,
        normalizer_path=NORMALIZER_PATH,
        pretrained_vision_language_model_name_or_path=RDT2_VQ_PATH,
        device=DEVICE,
        dtype=torch.bfloat16,  # 推荐使用 bfloat16 以匹配训练精度
    )
    
    # -------------------------------------------------------------------------
    # 3. 构造输入数据 (适配 3 相机 + 14 维 State)
    # -------------------------------------------------------------------------
    # 构造伪造图像 (384x384)
    # 在真实部署中，这里应该替换为 cv2.VideoCapture 读取的真实画面
    fake_img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    
    # [关键修改] Keys 必须与 post_train.yaml 中的 camera_names 一致
    observations = {
        'images': {
            'cam_high': fake_img,        # 对应 camera_names[0]
            'cam_left_wrist': fake_img,  # 对应 camera_names[1]
            'cam_right_wrist': fake_img, # 对应 camera_names[2]
        },
        # 14维 Proprioception State (关节角度 + 夹爪)
        'state': np.random.randn(14).astype(np.float32) 
    }
    
    # 语言指令
    instruction = "Pick up the apple."
    
    # -------------------------------------------------------------------------
    # 4. 执行推理
    # -------------------------------------------------------------------------
    print(f"正在执行推理，指令: '{instruction}' ...")
    
    # model.step() 内部流程:
    # 1. 编码图片和文本
    # 2. 运行 Flow Matching 降噪 (因为是 FM 模型)
    # 3. 反归一化 (Unnormalize)
    # 返回: (T, 14) 的 Tensor
    action_chunk = model.step(
        observations=observations,
        instruction=instruction
    )
    
    # -------------------------------------------------------------------------
    # 5. 结果处理
    # -------------------------------------------------------------------------
    if isinstance(action_chunk, torch.Tensor):
        action_chunk = action_chunk.detach().cpu().numpy()
        
    print(f"推理完成! 输出动作块形状: {action_chunk.shape}")
    # 预期输出: (24, 14) 
    # 24: action_chunk_size
    # 14: 双臂关节维度 (例如: 7 left + 7 right)
    
    # [关键修改] 移除 UMI 特有的夹爪重缩放逻辑
    # 因为您的数据是关节空间，第 9 和 19 维可能不存在或含义不同
    # 您的数据已经是物理空间的关节角度（因为经过了 unnormalize）
    
    # 简单打印结果
    print("\n前 5 帧动作数据 (示例):")
    # 假设前 7 维是左臂/右臂，打印前 7 维
    with np.printoptions(precision=4, suppress=True):
        print(action_chunk[:5, :7]) 

    print("\n检查数值范围:")
    print(f"Min: {action_chunk.min():.4f}, Max: {action_chunk.max():.4f}")

if __name__ == "__main__":
    main()