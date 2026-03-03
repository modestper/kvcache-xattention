export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"

tasks="qasper"

#qwen
# OUT_PATH="/home/chioe/project/sparq/efficiency/throughput-results/qwen.jsonl"
# MODEL_PATH="/DATA/models/Qwen2.5-7B-Instruct"
# MODEL_NAME="Qwen2.5-7B-Instruct"

#llama
OUT_PATH="/home/chioe/project/x-attention/efficiency/recall"
MODEL_PATH="/DATA/models/Llama-3.1-8B-Instruct"
MODEL_NAME="Llama-3.1-8B-Instruct"

PYTHONPATH="$PWD" python -u /home/chioe/project/x-attention/eval/TTFT-qasper/ttft_pred.py \
    --model_path ${MODEL_PATH} \
    --config_path "/home/chioe/project/x-attention/eval/LongBench/config"\
    --dataset_path "/home/chioe/project/recall-dataset/qasper.jsonl" \
    --output_dir ${OUT_PATH}   --model "Llama-3.1-8B-Instruct"\
    --model_name ${MODEL_NAME} --task "qasper" \
    --method "xattn" --stride 16  --type "recall"         # 关键：SparseQ





