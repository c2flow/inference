#!/bin/bash
# Unified accuracy test script for Mixtral-8x7B
# Supports both transformers and vllm modes with cache support
# Usage: ./run_accuracy.sh [BATCH_SIZE] [USE_VLLM] [TENSOR_PARALLEL_SIZE] [USE_CACHE]
#   BATCH_SIZE: batch size (default: 1)
#   USE_VLLM: 0 for transformers, 1 for vllm (default: 1)
#   TENSOR_PARALLEL_SIZE: tensor parallel size for vllm (default: 8)
#   USE_CACHE: 0 for fresh run, 1 for use cached outputs (default: 0)

# Set VLLM_WORKER_MULTIPROC_METHOD to spawn to avoid CUDA error
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

MLCOMMONS_ALL_PATH="$(dirname "$(dirname "$(dirname "$PWD")")")"

# Set NLTK_DATA and HF_HOME
export NLTK_DATA="${MLCOMMONS_ALL_PATH}/nltk_data"
export HF_HOME="${MLCOMMONS_ALL_PATH}/huggingface"

# Set CHECKPOINT_PATH, DATASET_PATH
CHECKPOINT_PATH="${MLCOMMONS_ALL_PATH}/model/Mixtral-8x7B-Instruct-v0.1"
DATASET_PATH="${MLCOMMONS_ALL_PATH}/dataset/09292024_mixtral_15k_mintoken2_v1.pkl"

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

# Parse arguments
BATCH_SIZE=${1:-1}
USE_VLLM=${2:-1}  # 0 = transformers, 1 = vllm (default: vllm)
TENSOR_PARALLEL_SIZE=${3:-8}
USE_CACHE=${4:-0}  # 0 = fresh run, 1 = use cached outputs

# Set output directory based on mode
if [ "$USE_VLLM" = "1" ]; then
    MODE="vllm"
    OUTPUT_LOG_DIR="output_accuracy_bs${BATCH_SIZE}_tp${TENSOR_PARALLEL_SIZE}_vllm"
else
    MODE="transformers"
    OUTPUT_LOG_DIR="output_accuracy_bs${BATCH_SIZE}_transformers"
fi

# Add cache indicator to output directory if using cache
if [ "$USE_CACHE" = "1" ]; then
    OUTPUT_LOG_DIR="${OUTPUT_LOG_DIR}_cached"
fi

# Create output log directory
mkdir -p ${OUTPUT_LOG_DIR}
mkdir -p "run_outputs"  # For cache files

echo "=== Mixtral-8x7B Accuracy Test ==="
echo "Mode: ${MODE}"
echo "Batch size: ${BATCH_SIZE}"
if [ "$USE_VLLM" = "1" ]; then
    echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
fi
if [ "$USE_CACHE" = "1" ]; then
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
        --batch-size ${BATCH_SIZE} \
        --dtype float32 \
        --device cuda:0"

# Add vllm-specific arguments if enabled
if [ "$USE_VLLM" = "1" ]; then
    CMD_ARGS="${CMD_ARGS} --vllm --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --num-workers 1"
fi

# Add cache argument if enabled
if [ "$USE_CACHE" = "1" ]; then
    CMD_ARGS="${CMD_ARGS} --use-cached-outputs"
fi

# Run the accuracy benchmark
echo "Starting accuracy benchmark..."
echo "Command: python3 -u main.py ${CMD_ARGS}"
echo ""

python3 -u main.py ${CMD_ARGS} 2>&1 | tee ${OUTPUT_LOG_DIR}/accuracy_${MODE}.log

echo ""
echo "=== Accuracy Benchmark Completed ==="
echo "Results saved to: ${OUTPUT_LOG_DIR}/"
echo "Log file: ${OUTPUT_LOG_DIR}/accuracy_${MODE}.log"
echo ""

# Evaluate accuracy if mlperf_log_accuracy.json exists
ACCURACY_LOG_FILE="${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json"
if [ -e "${ACCURACY_LOG_FILE}" ]; then
    echo "=== Evaluating Accuracy Results ==="


    python3 evaluate-accuracy.py \
        --checkpoint-path ${CHECKPOINT_PATH} \
        --mlperf-accuracy-file ${ACCURACY_LOG_FILE} \
        --dataset-file ${DATASET_PATH} \
        --dtype int32 2>&1 | tee ${OUTPUT_LOG_DIR}/accuracy_evaluation.log


    echo ""
    echo "=== Accuracy Evaluation Summary ==="
    if [ -f "${OUTPUT_LOG_DIR}/accuracy_evaluation.log" ]; then
        tail -20 "${OUTPUT_LOG_DIR}/accuracy_evaluation.log"
    fi
else
    echo "Warning: mlperf_log_accuracy.json not found in ${OUTPUT_LOG_DIR}/"
    echo "Accuracy evaluation skipped."
fi

echo ""
echo "=== Test Completed ==="
echo "All results saved to: ${OUTPUT_LOG_DIR}/"
echo ""
echo "Next steps:"
echo "1. Check accuracy scores in ${OUTPUT_LOG_DIR}/accuracy_evaluation.log"
echo "2. For subsequent runs with cache: ./run_accuracy.sh ${BATCH_SIZE} ${USE_VLLM} ${TENSOR_PARALLEL_SIZE} 1"
echo "3. Compare with reference scores in README.md"
