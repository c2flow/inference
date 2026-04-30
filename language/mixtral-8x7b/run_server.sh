#!/bin/bash
# Unified server test script for Mixtral-8x7B (vllm mode)
# Usage: ./run_server.sh [options]
#   --device, -d DEVICE             Device to use (default: cuda)
#   --fp8                           Use FP8 quantized model
#   --tensor-parallel-size, -tp N   Tensor parallel size (default: 8)
#   --data-parallel-size, -dp N     Data parallel size (default: 1)
#   --target-qps, -qps QPS         Target QPS (default: 10)
#   --executor-backend, -backend B  Executor backend (default: mp)

# Set VLLM_WORKER_MULTIPROC_METHOD to spawn to avoid CUDA error
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

MLCOMMONS_ALL_PATH="$(dirname "$(dirname "$(dirname "$PWD")")")"

# Set NLTK_DATA and HF_HOME
export NLTK_DATA="${MLCOMMONS_ALL_PATH}/nltk_data"
export HF_HOME="${MLCOMMONS_ALL_PATH}/huggingface"

# Set CHECKPOINT_PATH, DATASET_PATH
CHECKPOINT_PATH="${MLCOMMONS_ALL_PATH}/model/Mixtral-8x7B-Instruct-v0.1"
DATASET_PATH="${MLCOMMONS_ALL_PATH}/dataset/09292024_mixtral_15k_mintoken2_v1.pkl"

# Parse arguments
DEVICE="cuda"
FP8_MODE=false
TENSOR_PARALLEL_SIZE=8
DATA_PARALLEL_SIZE=1
TARGET_QPS=10
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
DTYPE="bfloat16"
BLOCK_SIZE=""
GPU_MEMORY_UTILIZATION=0.9
case "$DEVICE" in
    gcu)
        export TORCH_ECCL_AVOID_RECORD_STREAMS=true
        export VLLM_USE_V1=0
        export VLLM_ATTENTION_BACKEND=XFORMERS
        DTYPE="float16"
        BLOCK_SIZE="64"
        GPU_MEMORY_UTILIZATION=0.5
        ;;
esac

# FP8 overrides
if $FP8_MODE; then
    CHECKPOINT_PATH="${MLCOMMONS_ALL_PATH}/model/Mixtral-8x7B-Instruct-v0.1-FP8"
    DTYPE="auto"
fi

# Verify paths exist
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "Error: CHECKPOINT_PATH does not exist: ${CHECKPOINT_PATH}"
    echo "Please check MLCOMMONS_ALL_PATH: ${MLCOMMONS_ALL_PATH}"
    exit 1
fi

if [ ! -f "${DATASET_PATH}" ]; then
    echo "Error: DATASET_PATH does not exist: ${DATASET_PATH}"
    echo "Please check MLCOMMONS_ALL_PATH: ${MLCOMMONS_ALL_PATH}"
    exit 1
fi

# Set output directory
FP8_PREFIX=""
if $FP8_MODE; then
    FP8_PREFIX="fp8_"
fi
OUTPUT_LOG_DIR="output_server/${FP8_PREFIX}tp${TENSOR_PARALLEL_SIZE}_dp${DATA_PARALLEL_SIZE}_qps${TARGET_QPS}"

# Create output log directory
mkdir -p ${OUTPUT_LOG_DIR}

echo "=== Mixtral-8x7B Server Test ==="
echo "TP: ${TENSOR_PARALLEL_SIZE}, DP: ${DATA_PARALLEL_SIZE}"
echo "Target QPS: ${TARGET_QPS}"
echo "Device: ${DEVICE}, Dtype: ${DTYPE}"
echo "Dataset: $(basename ${DATASET_PATH})"
echo "Output directory: ${OUTPUT_LOG_DIR}"
echo ""

# Build command arguments
CMD_ARGS="--scenario Server \
        --model-path ${CHECKPOINT_PATH} \
        --user-conf user.conf \
        --total-sample-count 15000 \
        --dataset-path ${DATASET_PATH} \
        --output-log-dir ${OUTPUT_LOG_DIR} \
        --batch-size 1 \
        --dtype ${DTYPE} \
        --target-qps ${TARGET_QPS} \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --data-parallel-size ${DATA_PARALLEL_SIZE} \
        --distributed-executor-backend ${EXECUTOR_BACKEND} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --vllm --num-workers 1"

# Add block-size if specified
if [[ -n "$BLOCK_SIZE" ]]; then
    CMD_ARGS="${CMD_ARGS} --block-size ${BLOCK_SIZE}"
fi

# Run the server benchmark
echo "Starting server benchmark..."
echo "Command: python3 -u main.py ${CMD_ARGS}"
echo ""

python3 -u main.py ${CMD_ARGS} 2>&1 | tee ${OUTPUT_LOG_DIR}/server.log

echo ""
echo "=== Server Benchmark Completed ==="
echo "Results saved to: ${OUTPUT_LOG_DIR}/"
echo "Log file: ${OUTPUT_LOG_DIR}/server.log"
