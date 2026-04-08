# Set VLLM_WORKER_MULTIPROC_METHOD to spawn to avoid CUDA error
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

MLCOMMONS_ALL_PATH="$(dirname "$(dirname "$(dirname "$PWD")")")"

# Set NLTK_DATA and HF_HOME
export NLTK_DATA="${MLCOMMONS_ALL_PATH}/nltk_data"
export HF_HOME="${MLCOMMONS_ALL_PATH}/huggingface"

# Parse arguments
DEVICE="cuda"
FP8_MODE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --device requires a value" >&2
                exit 1
            fi
            DEVICE="$2"; shift 2 ;;
        --fp8)
            FP8_MODE=true; shift ;;
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

# Log file for overall execution
EXECUTION_LOG="execution_summary.log"
echo "Accuracy test execution started at $(date)" > "$EXECUTION_LOG"

# Arrays for parameters to iterate over
declare -a GPU_COUNTS=(1 2 4 8)

BASE_LOG_DIR="output_accuracy_offline"
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=4096
GPU_MEMORY_UTILIZATION=0.95
PIPELINE_PARALLEL_SIZE=1

# Iterate through all combinations
for gpu_count in "${GPU_COUNTS[@]}"; do
    echo "========================================"
    echo "Running experiment with GPU_COUNT=$gpu_count"
    echo "========================================"

    # Create unique output directory for this experiment
    if [[ -n "$BLOCK_SIZE" ]]; then
        EXP_LOG_DIR="${BASE_LOG_DIR}/exp__${LOG_PREFIX}tp_${gpu_count}_pp_${PIPELINE_PARALLEL_SIZE}_${BLOCK_SIZE}_${MAX_MODEL_LEN}_${MAX_NUM_BATCHED_TOKENS}_${GPU_MEMORY_UTILIZATION}"
    else
        EXP_LOG_DIR="${BASE_LOG_DIR}/exp__${LOG_PREFIX}tp_${gpu_count}_pp_${PIPELINE_PARALLEL_SIZE}_${MAX_MODEL_LEN}_${MAX_NUM_BATCHED_TOKENS}_${GPU_MEMORY_UTILIZATION}"
    fi
    mkdir -p "${EXP_LOG_DIR}"

    # Build extra args
    EXTRA_ARGS=""
    if [[ -n "$BLOCK_SIZE" ]]; then
        EXTRA_ARGS="--block-size ${BLOCK_SIZE}"
    fi

    # Run the experiment with error handling
    {
        python3 -u main.py --scenario Offline \
            --model-path "${CHECKPOINT_PATH}" \
            --batch-size 13368 \
            --accuracy \
            --dtype "${DTYPE}" \
            --user-conf user.conf \
            --total-sample-count 13368 \
            --dataset-path "${DATASET_PATH}" \
            --output-log-dir "${EXP_LOG_DIR}" \
            --tensor-parallel-size "${gpu_count}" \
            --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --enable-chunked-prefill \
            --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            ${EXTRA_ARGS} \
            --vllm 2>&1 | tee "${EXP_LOG_DIR}/offline.log"

        # Check if main.py succeeded
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "Main script for GPU_COUNT=$gpu_count SUCCEEDED"

            # Run accuracy evaluation
            python evaluation.py \
                --mlperf-accuracy-file "${EXP_LOG_DIR}/mlperf_log_accuracy.json" \
                --model-name "${CHECKPOINT_PATH}" \
                --dataset-file "${DATASET_PATH}" \
                --dtype int32 \
                2>&1 | tee "${EXP_LOG_DIR}/offline_accuracy.log"

            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                echo "Evaluation for GPU_COUNT=$gpu_count SUCCEEDED" | tee -a "$EXECUTION_LOG"
            else
                echo "Evaluation for GPU_COUNT=$gpu_count FAILED" | tee -a "$EXECUTION_LOG"
            fi
        else
            echo "Main script for GPU_COUNT=$gpu_count FAILED" | tee -a "$EXECUTION_LOG"
        fi
    }

    # Sleep for a short time to ensure proper cleanup between runs
    sleep 120
done

echo "Accuracy test execution completed at $(date)" >> "$EXECUTION_LOG"
echo "Summary of accuracy test:"
echo "-------------------------"
cat "$EXECUTION_LOG"
