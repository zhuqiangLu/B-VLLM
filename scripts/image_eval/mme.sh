#!/bin/bash


CKPT=""

CUDA_VISIBLE_DEVICES=0 python bvlm/image_eval/model_vqa_loader.py \
    --model-path ./work_dirs/$CKPT \
    --question-file data/evaluation/MME/llava_mme.jsonl \
    --image-folder data/evaluation/MME/MME_Benchmark_release_version \
    --answers-file data/evaluation/MME/answers/$CKPT.jsonl \
    --do_sample \
    --temperature 0.2 \
    --top_p 0.9 \

cd data/evaluation/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
