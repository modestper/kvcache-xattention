models="Llama-3.1-8B-Instruct"

methods="xattn"

tasks="samsum narrativeqa qasper triviaqa hotpotqa multifieldqa_en multifieldqa_zh 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"

for model in $models; do
    for task in $tasks; do
        for method in $methods; do
            python -u eval/LongBench/llama_pred.py \
             --model $model --task $task --method $method
        done
    done
done
