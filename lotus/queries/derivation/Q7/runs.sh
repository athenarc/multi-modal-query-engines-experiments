#!/bin/bash
sizes=(10 20 50)
# models_ollama=("gemma3:12b" "llama3.3:70b")
# models_vllm=("meta-llama/Llama-3.1-8B-Instruct")
models_dev=("RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8")
sizes_dev=(200)

dev="${1:-}"


if [ "$dev" != "dev" ]; then
    for size in "${sizes[@]}"; do
        for model in "${models_ollama[@]}"; do
            echo "Running Q7 with -s $size and m $model"
            python lotus/queries/derivation/Q7/map.py --wandb -s $size -m $model -p ollama
        done
    done

    for size in "${sizes[@]}"; do
        for model in "${models_vllm[@]}"; do
            echo "Running Q7 with -s $size and m $model"
            python lotus/queries/derivation/Q7/map.py --wandb -s $size -m $model -p vllm
            python lotus/queries/derivation/Q7/extract.py --wandb -s $size -m $model -p vllm
        done
    done
else
    for size in "${sizes_dev[@]}"; do
        for model in "${models_dev[@]}"; do
            echo "Running Q7 with -s $size and m $model"
            python lotus/queries/derivation/Q7/map.py --wandb -s $size -m $model -p vllm
        done
    done
fi