#!/bin/bash
# Unified accuracy test script for Mixtral-8x7B (vllm mode)
# Usage: ./run_accuracy.sh [options]
#   --device, -d DEVICE           Device to use (default: cuda)
#   --fp8                         Use FP8 quantized model
#   --dtype DTYPE                 Data type (default: bfloat16, GCU: float16)
#   --use-cached-outputs          Use cached outputs from previous runs
#   --tensor-parallel-size, -tp N Tensor parallel size (default: 8)

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
USE_CACHE=false
TENSOR_PARALLEL_SIZE=8
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
        --dtype)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --dtype requires a value" >&2
                exit 1
            fi
            DTYPE="$2"; shift 2 ;;
        --use-cached-outputs)
            USE_CACHE=true; shift ;;
        --tensor-parallel-size|-tp)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --tensor-parallel-size requires a value" >&2
                exit 1
            fi
            TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
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
OUTPUT_LOG_DIR="output_accuracy/${FP8_PREFIX}tp${TENSOR_PARALLEL_SIZE}_${DTYPE}_gpu${GPU_MEMORY_UTILIZATION}"

# Add cache indicator to output directory if using cache
if $USE_CACHE; then
    OUTPUT_LOG_DIR="${OUTPUT_LOG_DIR}_cached"
fi

# Create output log directory
mkdir -p ${OUTPUT_LOG_DIR}
mkdir -p "run_outputs"  # For cache files

echo "=== Mixtral-8x7B Accuracy Test ==="
echo "Batch size: 15000"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
if $USE_CACHE; then
    echo "Cache mode: Using cached outputs from run_outputs/"
else
    echo "Cache mode: Fresh run (will generate cache)"
fi
echo "Dataset: $(basename ${DATASET_PATH})"
echo "Output directory: ${OUTPUT_LOG_DIR}"
echo ""

# Build command arguments
CMD_ARGS="--scenario Offline \
        --model-path ${CHECKPOINT_PATH} \
        --accuracy \
        --user-conf user.conf \
        --total-sample-count 15000 \
        --dataset-path ${DATASET_PATH} \
        --output-log-dir ${OUTPUT_LOG_DIR} \
        --batch-size 15000 \
        --dtype ${DTYPE} \
        --vllm --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --num-workers 1 --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}"

# Add block-size if specified
if [[ -n "$BLOCK_SIZE" ]]; then
    CMD_ARGS="${CMD_ARGS} --block-size ${BLOCK_SIZE}"
fi

# Add cache argument if enabled
if $USE_CACHE; then
    CMD_ARGS="${CMD_ARGS} --use-cached-outputs"
fi

# Run the accuracy benchmark
echo "Starting accuracy benchmark..."
echo "Command: python3 -u main.py ${CMD_ARGS}"
echo ""

python3 -u main.py ${CMD_ARGS} 2>&1 | tee ${OUTPUT_LOG_DIR}/accuracy.log

echo ""
echo "=== Accuracy Benchmark Completed ==="
echo "Results saved to: ${OUTPUT_LOG_DIR}/"
echo "Log file: ${OUTPUT_LOG_DIR}/accuracy.log"
echo ""

# Evaluate accuracy if mlperf_log_accuracy.json exists
ACCURACY_LOG_FILE="${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json"
if [ -e "${ACCURACY_LOG_FILE}" ]; then
    echo "=== Evaluating Accuracy Results ==="

    python3 evaluate-accuracy.py \
        --checkpoint-path ${CHECKPOINT_PATH} \
        --mlperf-accuracy-file ${ACCURACY_LOG_FILE} \
        --dataset-file ${DATASET_PATH} \
        --dtype int64 2>&1 | tee ${OUTPUT_LOG_DIR}/accuracy_evaluation.log

else
    echo "Warning: mlperf_log_accuracy.json not found in ${OUTPUT_LOG_DIR}/"
    echo "Accuracy evaluation skipped."
fi

echo ""
echo "=== Test Completed ==="
echo "All results saved to: ${OUTPUT_LOG_DIR}/"
