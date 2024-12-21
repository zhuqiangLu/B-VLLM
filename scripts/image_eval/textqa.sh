#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=""

output_file=./data/evaluation/textvqa/answers/$SPLIT/$CKPT/merge.jsonl

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videollama2/image_eval/model_vqa_loader.py \
        --model-path ./work_dirs/$CKPT \
        --question-file ./data/evaluation/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ./data/evaluation/textvqa/train_images \
        --answers-file ./data/evaluation/textvqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 &
done

wait

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/evaluation/textvqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m videollama2.image_eval.eval_textvqa --annotation-file ./data/evaluation/textvqa/TextVQA_0.5.1_val.json  --result-file $output_file
