set -x

CUDA_VISIBLE_DEVICES=0,1

EVAL_DATA_DIR=eval
OUTPUT_DIR=eval_output
CKPT=""

CKPT_NAME=$(echo $CKPT | rev | cut -d'/' -f1 | rev)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 bvlm/eval/inference_video_oqa_msrvtt.py \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/MSVD_Zero_Shot_QA/video \
            --question-file ${EVAL_DATA_DIR}/MSVD_Zero_Shot_QA/test_qa.json \
            --output-file ${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi


API_KEY=""
python3 bvlm/eval/eval_video_oqa_activitynet.py \
    --pred-path ${output_file} \
    --output-dir ${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/gpt \
    --output-json ${output_file} \
    --api-key $API_KEY \
    --num-tasks 16
