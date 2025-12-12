#!/bin/bash
# Unified offline test script for Mixtral-8x7B
# Default: vllm mode with batch_size=1, tensor_parallel_size=8
# Usage: ./run_offline.sh [BATCH_SIZE] [USE_VLLM] [TENSOR_PARALLEL_SIZE]
#   BATCH_SIZE: batch size (default: 1)
#   USE_VLLM: 0 for transformers, 1 for vllm (default: 1)
#   TENSOR_PARALLEL_SIZE: tensor parallel size for vllm (default: 8)

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
BATCH_SIZE=${1:-1}
USE_VLLM=${2:-1}  # 0 = transformers, 1 = vllm (default: vllm)
TENSOR_PARALLEL_SIZE=${3:-8}

# Set output directory based on mode
if [ "$USE_VLLM" = "1" ]; then
    MODE="vllm"
    OUTPUT_LOG_DIR="output_offline_bs${BATCH_SIZE}_tp${TENSOR_PARALLEL_SIZE}_vllm"
else
    MODE="transformers"
    OUTPUT_LOG_DIR="output_offline_bs${BATCH_SIZE}_transformers"
fi

# Create output log directory
mkdir -p ${OUTPUT_LOG_DIR}

echo "=== Mixtral-8x7B Offline Test ==="
echo "Mode: ${MODE}"
echo "Batch size: ${BATCH_SIZE}"
if [ "$USE_VLLM" = "1" ]; then
    echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
fi
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
        --batch-size ${BATCH_SIZE} \
        --dtype float32 \
        --device cuda:0"

# Add vllm-specific arguments if enabled
if [ "$USE_VLLM" = "1" ]; then
    CMD_ARGS="${CMD_ARGS} --vllm --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --num-workers 1"
fi

# Run the benchmark
echo "Starting benchmark..."
python3 -u main.py ${CMD_ARGS} 2>&1 | tee ${OUTPUT_LOG_DIR}/offline_performance_${MODE}.log

echo ""
echo "=== Test Completed ==="
echo "Results saved to: ${OUTPUT_LOG_DIR}/"
echo "Log file: ${OUTPUT_LOG_DIR}/offline_performance_${MODE}.log"
