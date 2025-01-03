#!/bin/bash

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT=""

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videollama2/image_eval/model_vqa_loader.py \
    --model-path ./work_dirs/$CKPT \
    --question-file ./data/evaluation/vizwiz/llava_test.jsonl \
    --image-folder ./data/evaluation/vizwiz/test \
    --answers-file ./data/evaluation/vizwiz/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 &
done

wait

output_file=./data/evaluation/vizwiz/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/evaluation/vizwiz/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./data/evaluation/vizwiz/llava_test.jsonl \
    --result-file $output_file \
    --result-upload-file ./data/evaluation/vizwiz/answers_upload/$CKPT.json
