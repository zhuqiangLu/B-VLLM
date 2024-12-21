#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=""

output_file=./data/evaluation/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videollama2/image_eval/model_vqa_loader.py \
        --model-path ./work_dirs/$CKPT \
        --question-file ./data/evaluation/vqav2/$SPLIT.jsonl \
        --image-folder ./data/evaluation/vqav2/test2015 \
        --answers-file ./data/evaluation/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 &
done

wait


# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/evaluation/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --dir ./data/evaluation/vqav2 --split $SPLIT --ckpt $CKPT

