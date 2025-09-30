#!/bin/bash
sizes=(5 10 20)
models=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
    for model in "${models[@]}"; do
        echo "Evaluating with -s $size and -m $model"
        python evaluation/join/Q12/eval_scripts/lotus_q12_eval.py -s $size -m $model -p vllm
        python evaluation/join/Q12/eval_scripts/blendsql_q12_eval.py -s $size -m $model -p vllm
    done
done