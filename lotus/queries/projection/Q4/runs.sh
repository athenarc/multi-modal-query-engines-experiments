a#!/bin/bash
sizes=(100 200 300)
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Running Q4 with -s $size and m $model"
        python lotus/queries/selection/Q4/map.py --wandb -s $size -m $model -p ollama
    done
done

for size in "${sizes[@]}"; do
    for model in "${models_vllm[@]}"; do
        echo "Running Q4 with -s $size and m $model"
        python lotus/queries/selection/Q4/map.py --wandb -s $size -m $model -p vllm
    done
done