a#!/bin/bash
sizes=(50 100 200)
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Running Q2 with -s $size and m $model"
        python lotus/queries/selection/Q2/map.py --wandb -s $size -m $model -p ollama
    done
done

for size in "${sizes[@]}"; do
    for model in "${models_vllm[@]}"; do
        echo "Running Q2 with -s $size and m $model"
        python lotus/queries/selection/Q2/map.py --wandb -s $size -m $model -p vllm
        python lotus/queries/selection/Q2/extract.py --wandb -s $size -m $model -p vllm
    done
done