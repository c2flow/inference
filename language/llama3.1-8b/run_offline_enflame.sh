# Set VLLM_WORKER_MULTIPROC_METHOD to spawn to avoid CUDA error
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export TORCH_ECCL_AVOID_RECORD_STREAMS=true
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS

MLCOMMONS_ALL_PATH="$(dirname "$(dirname "$(dirname "$PWD")")")"

# Set NLTK_DATA and HF_HOME
export NLTK_DATA="${MLCOMMONS_ALL_PATH}/nltk_data"
export HF_HOME="${MLCOMMONS_ALL_PATH}/huggingface"

# Set CHECKPOINT_PATH, DATASET_PATH
CHECKPOINT_PATH="${MLCOMMONS_ALL_PATH}/model/Meta-Llama-3.1-8B-Instruct"
DATASET_PATH="${MLCOMMONS_ALL_PATH}/dataset/cnn_eval.json"

# Log file for overall execution
EXECUTION_LOG="execution_summary.log"
echo "Experiment execution started at $(date)" > "$EXECUTION_LOG"

# Arrays for parameters to iterate over
declare -a GPU_COUNTS=(1 2 4 8)
declare -a DTYPES=("bfloat16" "float16")
declare -a BATCH_SIZES=(1 4 16 64 256 1024)

BASE_LOG_DIR="output_offline"

# Iterate through all combinations
for gpu_count in "${GPU_COUNTS[@]}"; do
  for dtype in "${DTYPES[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do  # Add batch size loop
      echo "========================================"
      echo "Running experiment with GPU_COUNT=$gpu_count, DTYPE=$dtype, BATCH_SIZE=$batch_size"
      echo "========================================"
      
      # Create unique output directory for this experiment
      EXP_LOG_DIR="${BASE_LOG_DIR}/exp__tp_${gpu_count}__dtype_${dtype}__bs_${batch_size}"
      mkdir -p "${EXP_LOG_DIR}"
      
      # Run the experiment with error handling
      {
        python3 -u main.py --scenario Offline \
            --model-path "${CHECKPOINT_PATH}" \
            --batch-size "${batch_size}" \
            --dtype "${dtype}" \
            --user-conf user.conf \
            --total-sample-count 13368 \
            --dataset-path "${DATASET_PATH}" \
            --output-log-dir "${EXP_LOG_DIR}" \
            --tensor-parallel-size "${gpu_count}" \
            --max-model-len 16384 \
            --enable-chunked-prefill \
            --vllm 2>&1 | tee "${EXP_LOG_DIR}/offline.log"
        
        # Check if the experiment succeeded
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
          echo "Experiment with GPU_COUNT=$gpu_count, DTYPE=$dtype SUCCEEDED" | tee -a "$EXECUTION_LOG"
        else
          echo "Experiment with GPU_COUNT=$gpu_count, DTYPE=$dtype FAILED" | tee -a "$EXECUTION_LOG"
        fi
      }
      
      # Sleep for a short time to ensure proper cleanup between runs
      sleep 120
    done
  done
done

echo "Experiment execution completed at $(date)" >> "$EXECUTION_LOG"
echo "Summary of experiments:"
echo "----------------------"
cat "$EXECUTION_LOG"