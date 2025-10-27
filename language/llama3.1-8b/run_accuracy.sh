# Set VLLM_WORKER_MULTIPROC_METHOD to spawn to avoid CUDA error
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# Set CHECKPOINT_PATH, DATASET_PATH, OUTPUT_LOG_DIR, and GPU_COUNT
CHECKPOINT_PATH="${CHECKPOINT_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-cnn_eval.json}"
OUTPUT_LOG_DIR="${OUTPUT_LOG_DIR:-output}"
GPU_COUNT="${GPU_COUNT:-8}"

# Create output log directory
mkdir -p ${OUTPUT_LOG_DIR}

python3 -u main.py --scenario Offline \
        --model-path ${CHECKPOINT_PATH} \
        --batch-size 16 \
        --accuracy \
        --dtype bfloat16 \
        --user-conf user.conf \
        --total-sample-count 13368 \
        --dataset-path ${DATASET_PATH} \
        --output-log-dir ${OUTPUT_LOG_DIR} \
        --tensor-parallel-size ${GPU_COUNT} \
        --vllm 2>&1 | tee ${OUTPUT_LOG_DIR}/offline.log

python evaluation.py \
        --mlperf-accuracy-file ${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json \
        --model-name ${CHECKPOINT_PATH} \
        --dataset-file ${DATASET_PATH} \
        --dtype int32 \
        2>&1 | tee ${OUTPUT_LOG_DIR}/offline_accuracy.log