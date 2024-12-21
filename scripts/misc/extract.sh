set -x


CUDA_VISIBLE_DEVICES=0,1,2,3
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

for IDX in $(seq 0 $((CHUNKS-1))); do
    # select the GPUs for the task
    gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
    TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=${gpu_devices} python extract.py \
        --video_dir eval/videomme/videos \
        --feat_dir eval/videomme/feat \
        --fps 1 \
        --infer_batch 512 \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done