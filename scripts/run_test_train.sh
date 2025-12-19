#!/bin/bash

# 1. 定义任务名和配置文件路径
TASK="test-train-ur5e"
# 指向您刚刚修改的配置文件
DATASET_CONFIG_PATH="configs/datasets/test_train.yaml" 
export NCCL_P2P_DISABLE=1
# 2. 定义模型路径 (使用您推理时验证过的本地路径)
export TOKENIZER_ID="/home/jiangjiahao/data/model/Qwen2.5-VL-7B-Instruct"
export VAE_ID="/home/jiangjiahao/data/model/RVQActionTokenizer"
export MODEL_ID="/home/jiangjiahao/data/model/RDT2-VQ"

# 3. 输出目录
export OUTPUT_DIR="./outputs/vqvla-sft-${TASK}-lora"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# 4. 启动训练
# 建议先设置 CUDA_VISIBLE_DEVICES 指定显卡，例如使用您之前测试的 cuda:3
export CUDA_VISIBLE_DEVICES=2,3,4,5

# 注意：如果单卡运行，可以直接用 python main.py，或者配置 accelerate config
# 这里演示直接 python 运行以简化多卡配置问题，生产环境建议用 accelerate launch
accelerate launch main.py \
    --deepspeed="scripts/zero1.json" \
    --tokenizer_name=$TOKENIZER_ID \
    --vae_name=$VAE_ID \
    --pretrained_model_name_or_path=$MODEL_ID \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=32 \
    --eval_batch_size=4 \
    --max_train_steps=10000 \
    --eval_strategy="no" \
    --logging_steps=1 \
    --checkpoints_total_limit=20 \
    --checkpointing_step=1000 \
    --lr_scheduler="cosine" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=16 \
    --gradient_checkpointing \
    --log_level="info" \
    --report_to="wandb" \
    --lr_warmup_steps=500 \
    --dataset=$DATASET_CONFIG_PATH \
    --image_corruption \
    --use_default_collate_fn_for_eval