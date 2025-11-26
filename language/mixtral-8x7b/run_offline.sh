# Set VLLM_WORKER_MULTIPROC_METHOD to spawn to avoid CUDA error
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

MLCOMMONS_ALL_PATH="$(dirname "$(dirname "$(dirname "$PWD")")")"

# Set NLTK_DATA and HF_HOME
export NLTK_DATA="${MLCOMMONS_ALL_PATH}/nltk_data"
export HF_HOME="${MLCOMMONS_ALL_PATH}/huggingface"

# Set CHECKPOINT_PATH, DATASET_PATH
CHECKPOINT_PATH="${MLCOMMONS_ALL_PATH}/model/Mixtral-8x7B-Instruct-v0.1"
DATASET_PATH="${MLCOMMONS_ALL_PATH}/dataset/2024_06_06_mixtral_15k_v4.pkl"

# Set BATCH_SIZE and OUTPUT_LOG_DIR
BATCH_SIZE=16
OUTPUT_LOG_DIR="output_offline_bs${BATCH_SIZE}"

# Create output log directory
mkdir -p ${OUTPUT_LOG_DIR}

python3 -u main.py --scenario Offline \
        --model-path ${CHECKPOINT_PATH} \
        --user-conf user.conf \
        --total-sample-count 15000 \
        --dataset-path ${DATASET_PATH} \
        --output-log-dir ${OUTPUT_LOG_DIR} \
        --batch-size ${BATCH_SIZE} \
        --dtype float32 \
        --device cuda:0 2>&1 | tee ${OUTPUT_LOG_DIR}/offline_performance_log.log
