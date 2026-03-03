model=$1
task=$2
method=$3
python -u eval/LongBench/llama_pred.py \
    --model $model --task $task \
    --method $method
