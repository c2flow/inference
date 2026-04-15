#!/bin/bash
# Unified server test script for Llama3.1-8B (vllm mode)
# Usage: ./run_server.sh [options]
#   --device, -d DEVICE             Device to use (default: cuda)
#   --fp8                           Use FP8 quantized model
#   --tensor-parallel-size, -tp N   Tensor parallel size (default: 1)
#   --data-parallel-size, -dp N     Data parallel size (default: 1)
#   --pipeline-parallel-size, -pp N Pipeline parallel size (default: 1)
#   --target-qps, -qps QPS         Target QPS (default: 50)

# Set VLLM_WORKER_MULTIPROC_METHOD to spawn to avoid CUDA error
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

MLCOMMONS_ALL_PATH="$(dirname "$(dirname "$(dirname "$PWD")")")"

# Set NLTK_DATA and HF_HOME
export NLTK_DATA="${MLCOMMONS_ALL_PATH}/nltk_data"
export HF_HOME="${MLCOMMONS_ALL_PATH}/huggingface"

# Parse arguments
DEVICE="cuda"
FP8_MODE=false
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
TARGET_QPS=50
EXECUTOR_BACKEND="mp"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device|-d)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --device requires a value" >&2
                exit 1
            fi
            DEVICE="$2"; shift 2 ;;
        --fp8)
            FP8_MODE=true; shift ;;
        --tensor-parallel-size|-tp)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --tensor-parallel-size requires a value" >&2
                exit 1
            fi
            TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --data-parallel-size|-dp)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --data-parallel-size requires a value" >&2
                exit 1
            fi
            DATA_PARALLEL_SIZE="$2"; shift 2 ;;
        --pipeline-parallel-size|-pp)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --pipeline-parallel-size requires a value" >&2
                exit 1
            fi
            PIPELINE_PARALLEL_SIZE="$2"; shift 2 ;;
        --target-qps|-qps)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --target-qps requires a value" >&2
                exit 1
            fi
            TARGET_QPS="$2"; shift 2 ;;
        --executor-backend|-backend)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --executor-backend requires a value" >&2
                exit 1
            fi
            EXECUTOR_BACKEND="$2"; shift 2 ;;
        *)
            echo "Warning: Unknown option $1" >&2; shift ;;
    esac
done

# Device-specific defaults
DTYPE="auto"
BLOCK_SIZE=""
case "$DEVICE" in
    gcu)
        export TORCH_ECCL_AVOID_RECORD_STREAMS=true
        export VLLM_USE_V1=0
        export VLLM_ATTENTION_BACKEND=XFORMERS
        DTYPE="float16"
        BLOCK_SIZE="64"
        ;;
esac

# Set CHECKPOINT_PATH, DATASET_PATH
DATASET_PATH="${MLCOMMONS_ALL_PATH}/dataset/cnn_eval.json"
if $FP8_MODE; then
    CHECKPOINT_PATH="${MLCOMMONS_ALL_PATH}/model/Meta-Llama-3.1-8B-Instruct-FP8"
    LOG_PREFIX="fp8_"
else
    CHECKPOINT_PATH="${MLCOMMONS_ALL_PATH}/model/Meta-Llama-3.1-8B-Instruct"
    LOG_PREFIX=""
fi

# Log directory
BASE_LOG_DIR="output_server"
EXP_LOG_DIR="${BASE_LOG_DIR}/exp__${LOG_PREFIX}tp${TENSOR_PARALLEL_SIZE}_dp${DATA_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}_qps${TARGET_QPS}"
mkdir -p "${EXP_LOG_DIR}"

# Fixed model parameters
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=4096
GPU_MEMORY_UTILIZATION=0.95

# Build extra args
EXTRA_ARGS=""
if [[ -n "$BLOCK_SIZE" ]]; then
    EXTRA_ARGS="--block-size ${BLOCK_SIZE}"
fi

echo "=== Llama3.1-8B Server Test ==="
echo "TP: ${TENSOR_PARALLEL_SIZE}, DP: ${DATA_PARALLEL_SIZE}, PP: ${PIPELINE_PARALLEL_SIZE}"
echo "Target QPS: ${TARGET_QPS}"
echo "Device: ${DEVICE}, Dtype: ${DTYPE}"
echo "Output: ${EXP_LOG_DIR}"
echo ""

# Run the benchmark
python3 -u main.py --scenario Server \
    --model-path "${CHECKPOINT_PATH}" \
    --batch-size 16 \
    --dtype "${DTYPE}" \
    --user-conf user.conf \
    --total-sample-count 13368 \
    --dataset-path "${DATASET_PATH}" \
    --output-log-dir "${EXP_LOG_DIR}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --data-parallel-size "${DATA_PARALLEL_SIZE}" \
    --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-chunked-prefill \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --target-qps "${TARGET_QPS}" \
    --distributed-executor-backend "${EXECUTOR_BACKEND}" \
    ${EXTRA_ARGS} \
    --vllm 2>&1 | tee "${EXP_LOG_DIR}/server.log"

echo ""
echo "=== Test Completed ==="
echo "Results saved to: ${EXP_LOG_DIR}/"
