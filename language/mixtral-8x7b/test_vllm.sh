#!/bin/bash
# Test script for Mixtral-8x7B vllm mode
# Usage: ./test_vllm.sh [BATCH_SIZE] [TENSOR_PARALLEL_SIZE]

# Set VLLM_WORKER_MULTIPROC_METHOD to spawn to avoid CUDA error
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

MLCOMMONS_ALL_PATH="$(dirname "$(dirname "$(dirname "$PWD")")")"

# Set NLTK_DATA and HF_HOME
export NLTK_DATA="${MLCOMMONS_ALL_PATH}/nltk_data"
export HF_HOME="${MLCOMMONS_ALL_PATH}/huggingface"

# Set CHECKPOINT_PATH, DATASET_PATH
CHECKPOINT_PATH="${MLCOMMONS_ALL_PATH}/model/Mixtral-8x7B-Instruct-v0.1"
DATASET_PATH="${MLCOMMONS_ALL_PATH}/dataset/09292024_mixtral_15k_mintoken2_v1.pkl"

# Set BATCH_SIZE and OUTPUT_LOG_DIR
BATCH_SIZE=${1:-1}
TENSOR_PARALLEL_SIZE=${2:-8}
OUTPUT_LOG_DIR="output_vllm_offline_bs${BATCH_SIZE}_tp${TENSOR_PARALLEL_SIZE}"

# Create output log directory
mkdir -p ${OUTPUT_LOG_DIR}

echo "Testing Mixtral-8x7B with vllm mode"
echo "Batch size: ${BATCH_SIZE}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "Output directory: ${OUTPUT_LOG_DIR}"
echo ""

python3 -u main.py --scenario Offline \
        --model-path ${CHECKPOINT_PATH} \
        --user-conf user.conf \
        --total-sample-count 15000 \
        --dataset-path ${DATASET_PATH} \
        --output-log-dir ${OUTPUT_LOG_DIR} \
        --batch-size ${BATCH_SIZE} \
        --dtype float32 \
        --vllm \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --num-workers 1 2>&1 | tee ${OUTPUT_LOG_DIR}/vllm_performance_log.log

echo ""
echo "Test completed. Check logs in ${OUTPUT_LOG_DIR}/"