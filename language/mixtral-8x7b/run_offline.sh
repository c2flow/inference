#!/bin/bash
# Unified offline test script for Mixtral-8x7B (vllm mode)
# Usage: ./run_offline.sh [options]
#   --device, -d DEVICE           Device to use (default: cuda)
#   --fp8                         Use FP8 quantized model
#   --dtype DTYPE                 Data type (default: bfloat16, GCU: float16)
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
        ;;
esac

# FP8 overrides
if $FP8_MODE; then
    CHECKPOINT_PATH="${MLCOMMONS_ALL_PATH}/model/Mixtral-8x7B-Instruct-v0.1-FP8"
    DTYPE="auto"
fi

# Set output directory
FP8_PREFIX=""
if $FP8_MODE; then
    FP8_PREFIX="fp8_"
fi
OUTPUT_LOG_DIR="output_offline/${FP8_PREFIX}tp${TENSOR_PARALLEL_SIZE}_${DTYPE}_gpu${GPU_MEMORY_UTILIZATION}"

# Create output log directory
mkdir -p ${OUTPUT_LOG_DIR}

echo "=== Mixtral-8x7B Offline Test ==="
echo "Batch size: 15000"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "Dataset: $(basename ${DATASET_PATH})"
echo "Output directory: ${OUTPUT_LOG_DIR}"
echo ""

# Build command arguments
CMD_ARGS="--scenario Offline \
        --model-path ${CHECKPOINT_PATH} \
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

# Run the benchmark
echo "Starting benchmark..."
python3 -u main.py ${CMD_ARGS} 2>&1 | tee ${OUTPUT_LOG_DIR}/offline_performance.log

echo ""
echo "=== Test Completed ==="
echo "Results saved to: ${OUTPUT_LOG_DIR}/"
echo "Log file: ${OUTPUT_LOG_DIR}/offline_performance.log"
