# Set VLLM_WORKER_MULTIPROC_METHOD to spawn to avoid CUDA error
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

MLCOMMONS_ALL_PATH="$(dirname "$(dirname "$(dirname "$PWD")")")"

# Set NLTK_DATA and HF_HOME
export NLTK_DATA="${MLCOMMONS_ALL_PATH}/nltk_data"
export HF_HOME="${MLCOMMONS_ALL_PATH}/huggingface"

# Set CHECKPOINT_PATH, DATASET_PATH
CHECKPOINT_PATH="${MLCOMMONS_ALL_PATH}/model/Meta-Llama-3.1-8B-Instruct-FP8"
DATASET_PATH="${MLCOMMONS_ALL_PATH}/dataset/cnn_eval.json"

# Log file for overall execution
EXECUTION_LOG="execution_summary.log"
echo "Accuracy test execution started at $(date)" > "$EXECUTION_LOG"

BASE_LOG_DIR="output_accuracy_offline"
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=4096
GPU_MEMORY_UTILIZATION=0.95
GPU_COUNT=1

# Create unique output directory for this experiment
EXP_LOG_DIR="${BASE_LOG_DIR}/exp__fp8_tp_${GPU_COUNT}_${MAX_MODEL_LEN}_${MAX_NUM_BATCHED_TOKENS}_${GPU_MEMORY_UTILIZATION}"
mkdir -p "${EXP_LOG_DIR}"

# Run the accuracy test with error handling
{
python3 -u main.py --scenario Offline \
    --model-path "${CHECKPOINT_PATH}" \
    --batch-size 13368 \
    --accuracy \
    --dtype auto \
    --user-conf user.conf \
    --total-sample-count 13368 \
    --dataset-path "${DATASET_PATH}" \
    --output-log-dir "${EXP_LOG_DIR}" \
    --tensor-parallel-size "${GPU_COUNT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-chunked-prefill \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --vllm 2>&1 | tee "${EXP_LOG_DIR}/offline.log"
}

# Run accuracy evaluation
python evaluation.py \
    --mlperf-accuracy-file "${EXP_LOG_DIR}/mlperf_log_accuracy.json" \
    --model-name "${CHECKPOINT_PATH}" \
    --dataset-file "${DATASET_PATH}" \
    --dtype int32 \
    2>&1 | tee "${EXP_LOG_DIR}/offline_accuracy.log"

echo "Accuracy test execution completed at $(date)" >> "$EXECUTION_LOG"
echo "Summary of accuracy test:"
echo "-------------------------"
cat "$EXECUTION_LOG"