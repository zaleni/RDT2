# Define your env settings here 
# e.g., nccl, network, proxy, etc.

# set your own CFLAGS and LDFLAGS here
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=2,3,4,5
CUR_TIME=$(date +%Y%m%d_%H%M%S)

ENV_TYPE="robotwin_train"   # 改这个是加载rdt config的
WANDB_PROJECT="rdt2-action-expert-robotwin-full"
OUTPUT_DIR="/home/jiangjiahao/data/ckpt/RDT2/rdt/${WANDB_PROJECT}/"


mkdir -p "./logs/${WANDB_PROJECT}"
LOGGING_DIR="./logs/${WANDB_PROJECT}"
LOGGING_FILE="${LOGGING_DIR}/${CUR_TIME}.log"

WDS_CONFIG_FILE="configs/datasets/robotwin.yaml"  # 改这个是加载webdataset config的
VISION_LANGUAGE_MODEL_NAME_OR_PATH="/home/jiangjiahao/data/model/RDT2-VQ"

TRAIN_BATCH_SIZE=4
SAMPLE_BATCH_SIZE=2

# delete if exists
if [ -f "$LOGGING_FILE" ]; then
    rm "$LOGGING_FILE"
    echo "Log file '$LOGGING_FILE' deleted"
else
    echo "Log file '$LOGGING_FILE' does not exist"
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# e.g. PYTHONPATH="/ssd/RDT2"
PYTHONPATH="/home/jiangjiahao/research/RDT2" accelerate launch rdt/main.py \
    --deepspeed="scripts/zero1.json" \
    --config_path="./configs/rdt/${ENV_TYPE}.yaml" \
    --train_vlm \
    --vlm_learning_rate=1e-5 \
    --learning_rate=1e-4 \
    --vlm_weight_decay=0.0 \
    --pretrained_vision_language_model_name_or_path=$VISION_LANGUAGE_MODEL_NAME_OR_PATH \
    --output_dir=$OUTPUT_DIR \
    --webdataset_config=$WDS_CONFIG_FILE \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --sample_batch_size=$SAMPLE_BATCH_SIZE \
    --gradient_accumulation_steps=1 \
    --max_train_steps=30000 \
    --checkpointing_period=5000 \
    --sample_period=100000 \
    --checkpoints_total_limit=10 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=500 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --report_to=wandb 2>&1 | tee -a "$LOGGING_FILE"
