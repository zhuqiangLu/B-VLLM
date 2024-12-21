#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


CKPT=""

output_file=./work_dirs/pope/answers/$CKPT/merge.jsonl
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videollama2/image_eval/model_vqa_loader.py \
        --model-path work_dirs/$CKPT \
        --question-file ./data/evaluation/pope/llava_pope_test.jsonl \
        --image-folder ./data/evaluation/pope/val2014 \
        --answers-file ./work_dirs/pope/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 &
done

wait

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./work_dirs/pope/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python videollama2/image_eval/eval_pope.py \
    --annotation-dir ./data/evaluation/pope/coco \
    --question-file ./data/evaluation/pope/llava_pope_test.jsonl \
    --result-file $output_file
