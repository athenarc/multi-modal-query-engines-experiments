#!/bin/bash
sizes=(10 20 50)
# models_ollama=("gemma3:12b" "llama3.3:70b")
# models_vllm=("meta-llama/Llama-3.1-8B-Instruct")
models_dev=("gemma3:12b")
sizes_dev=(10)

dev="${1:-}"


if [ "$dev" != "dev" ]; then
    for size in "${sizes[@]}"; do
        for model in "${models_ollama[@]}"; do
            echo "Running Q2 with -s $size and m $model"
            python lotus/queries/derivation/Q2/map.py --wandb -s $size -m $model -p ollama
        done
    done

    for size in "${sizes[@]}"; do
        for model in "${models_vllm[@]}"; do
            echo "Running Q2 with -s $size and m $model"
            python lotus/queries/derivation/Q2/map.py --wandb -s $size -m $model -p vllm
            python lotus/queries/derivation/Q2/extract.py --wandb -s $size -m $model -p vllm
        done
    done
else
    for size in "${sizes_dev[@]}"; do
        for model in "${models_dev[@]}"; do
            echo "Running Q1 with -s $size and m $model"
            python lotus/queries/derivation/Q1/map.py  -s $size -m $model -p ollama
            echo "Evaluating Q1 with -s $size and m $model executed from Lotus"
            python evaluation/derivation/Q1/eval_scripts/lotus_q1_eval.py  -s $size -m $model -p ollama
        done
    done
fi