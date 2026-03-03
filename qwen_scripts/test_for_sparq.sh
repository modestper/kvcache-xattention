
# Root Directories
GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR="benchmark_root" # the path that stores generated task samples and model predictions.
MODEL_DIR="/home/chioe/models" # the path that contains individual model folders from HUggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization

cd eval/RULER/scripts
# Model and Tokenizer
source config_models.sh

#MODEL_NAME="Llama-3.1-8B-Instruct"
MODEL_NAME="Qwen2.5-7B-Instruct"

MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"


export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}

# Benchmark and Tasks
source config_tasks.sh
BENCHMARK="synthetic"
#declare -n TASKS=$BENCHMARK
#TASKS="niah_single_1" "niah_single_2" "niah_single_3" "niah_multikey_1" "niah_multikey_2" "niah_multikey_3" "niah_multivalue" "niah_multiquery" "vt" "cwe" "fwe" "qa_1" "qa_2"
TASKS="niah_single_3"
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred02"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in "${TASKS[@]}"; do
        start_time=$(date +%s)
        python pred/qwen-call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type "hf" \
            --model_name_or_path "/DATA/models/Qwen2.5-7B-Instruct" \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            --batch_size ${BATCH_SIZE} \
            ${STOP_WORDS}
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
    done
    
     python eval/evaluate.py \
         --data_dir ${PRED_DIR} \
         --benchmark ${BENCHMARK}
done

echo "Total time spent on call_api: $total_time seconds"
