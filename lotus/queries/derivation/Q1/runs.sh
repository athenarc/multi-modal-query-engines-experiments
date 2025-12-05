#!/bin/bash
sizes=(10 20 50)
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")
# models_dev=("gemma3:12b")
# models_dev=("RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8")
models_dev=("Qwen/Qwen3-8B")
sizes_dev=(200)

dev="${1:-}"


if [ "$dev" != "dev" ]; then
    for size in "${sizes[@]}"; do
        for model in "${models_ollama[@]}"; do
            echo "Running Q1 with -s $size and m $model"
            python lotus/queries/derivation/Q1/map.py --wandb -s $size -m $model -p ollama
        done
    done

    for size in "${sizes[@]}"; do
        for model in "${models_vllm[@]}"; do
            echo "Running Q1 with -s $size and m $model"
            python lotus/queries/derivation/Q1/map.py --wandb -s $size -m $model -p vllm
            python lotus/queries/derivation/Q1/extract.py --wandb -s $size -m $model -p vllm
        done
    done
else
    for size in "${sizes_dev[@]}"; do
        for model in "${models_dev[@]}"; do
            echo "Running Lotus-map, Q1 with -s $size and m $model"
            python lotus/queries/derivation/Q1/map.py  -s $size -m $model -p vllm
            echo "Evaluating Lotus-map, Q1 with -s $size and m $model executed from Lotus"
            python evaluation/derivation/Q1/eval_scripts/lotus_q1_eval.py  -s $size -m $model -p vllm

            echo "Running Lotus-extract, Q1 with -s $size and m $model"
            python lotus/queries/derivation/Q1/extract.py  -s $size -m $model -p vllm
            echo "Evaluating Lotus-extract, Q1 with -s $size and m $model executed from Lotus"
            python evaluation/derivation/Q1/eval_scripts/lotus_q1_eval.py  -e -s $size -m $model -p vllm
        done
    done
fi