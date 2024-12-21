#!/bin/bash

SPLIT="mmbench_dev_20230712"
CKPT=""

CUDA_VISIBLE_DEVICES=0 python videollama2/image_eval/model_vqa_mmbench.py \
    --model-path ./work_dirs/$CKPT \
    --question-file ./data/evaluation/mmbench/$SPLIT.tsv \
    --answers-file ./data/evaluation/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt 

mkdir -p ./data/evaluation/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./data/evaluation/mmbench/$SPLIT.tsv \
    --result-dir ./data/evaluation/mmbench/answers/$SPLIT \
    --upload-dir ./data/evaluation/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
