
TEMPERATURE="0.0" # greedy
TOP_P="1.0"
TOP_K="32"

SEQ_LENGTHS=(
    65536
    32768
    16384
    8192
    4096
)

MODEL_SELECT() {
    MODEL_NAME=$1
    MODEL_DIR=$2
    ENGINE_DIR=$3
    
    case $MODEL_NAME in
        # 原来的 llama3.1-8b-chat 配置（保留）
        llama3.1-8b-chat)
            MODEL_PATH="${MODEL_DIR}/Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            MODEL_FRAMEWORK="hf"
            ;;

        # 新增：Qwen2.5-7B-Instruct
        Qwen2.5-7B-Instruct)
            # 本地模型目录：按你的实际路径改，如果不在 /DATA/models 就改这里
            MODEL_PATH="/DATA/models/Qwen2.5-7B-Instruct"

            # 模板类型：RULER 的 data/prepare.py 会根据这个选 prompt 模板
            # Qwen 用的是 ChatML 风格，这里填 chatml（大部分仓库都有这个分支）
            # MODEL_TEMPLATE_TYPE="chatml"
            MODEL_TEMPLATE_TYPE="meta-llama3"  # Qwen 结构类似 Llama，可以用这个模板

            # 用本地 HuggingFace + 你改好的 xattention（不是 vllm / trtllm / sglang）
            MODEL_FRAMEWORK="hf"
            ;;
    esac

    # 下面这段是通用逻辑：如果上面 case 里没手动设置 TOKENIZER_PATH，
    # 就根据模型目录自动推断是 Nemo 还是 HF 的 tokenizer
    if [ -z "${TOKENIZER_PATH}" ]; then
        if [ -f ${MODEL_PATH}/tokenizer.model ]; then
            TOKENIZER_PATH=${MODEL_PATH}/tokenizer.model
            TOKENIZER_TYPE="nemo"
        else
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
        fi
    fi

    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT"
}