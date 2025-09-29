#!/bin/bash
sizes=(20 50 100)
models=("gemma3:12b" "llama3.1:8b")

for size in "${sizes[@]}"; do
    for model in "${models[@]}"; do
        echo "Evaluating with -s $size and -m $model"
        python evaluation/join/Q11/eval_scripts/lotus_q11_eval.py -s $size -m $model -p ollama
        python evaluation/join/Q11/eval_scripts/blendsql_q11_eval.py -s $size -m $model -p ollama
    done
done