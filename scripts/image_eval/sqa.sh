#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}



CKPT=""

output_file=./work_dirs/scienceqa/answers/$CKPT/merge.jsonl
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python  videollama2/image_eval/model_vqa_science.py \
        --model-path work_dirs/$CKPT \
        --question-file data/evaluation/scienceqa/llava_test_CQM-A.json \
        --image-folder data/evaluation/scienceqa/images/test \
        --answers-file work_dirs/scienceqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt &
done

wait

> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./work_dirs/scienceqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python videollama2/image_eval/eval_science_qa.py \
    --base-dir data/evaluation/scienceqa \
    --result-file $output_file \
    --output-file work_dirs/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result work_dirs/scienceqa/answers/${CKPT}_result.json
