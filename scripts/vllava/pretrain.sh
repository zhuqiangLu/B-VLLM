#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments

# Log Arguments
# export TRANSFORMERS_OFFLINE=0
export WANDB_PROJECT=BVLM
RUN_NAME=vllava_settings_dynamic
DATA_DIR=datasets
OUTP_DIR=work_dirs

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE  \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    bvlm/train_flash_attn.py \
    --deepspeed scripts/zero2.json \
    --model_type videollama2_qwen2 \
    --model_path Qwen/Qwen2-7B-Instruct \
    --vision_tower ./model_zoo/LAVIS/eva_vit_g.pth \
    --pretrain_qformer model_zoo/LAVIS/instruct_blip_vicuna7b_trimmed.pth \
    --bert_type "qformer_pretrain_freeze" \
    --image_processor "openai/clip-vit-large-patch14-336" \
    --mm_projector_type spatial_conv \
    --tune_mm_mlp_adapter True \
    --data_path   ${DATA_DIR}/videollava_pt/valley_llavaimage.json \
    --data_folder ${DATA_DIR}/videollava_pt/ \
    --mm_vision_select_layer -2 \
    --num_frames 16 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/pretrain_${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 800 \
    --save_total_limit 3 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $RUN_NAME 