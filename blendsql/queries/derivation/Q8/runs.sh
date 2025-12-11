#!/bin/bash
sizes=(10 30 50)
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")
models_dev=("gemma3:12b")
# models_dev=("RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8")
# models_dev=("Qwen/Qwen3-8B")
sizes_dev=(50 100 200 300 500)

dev="${1:-}"


if [ "$dev" != "dev" ]; then
    for size in "${sizes[@]}"; do
        for model in "${models_ollama[@]}"; do
            echo "Running Q8 with -s $size and m $model"
            python blendsql/queries/derivation/Q8/q8.py --wandb -s $size -m $model -p ollama
        done
    done

    for size in "${sizes[@]}"; do
        for model in "${models_vllm[@]}"; do
            echo "Running Q8 with -s $size and m $model"
            python blendsql/queries/derivation/Q8/q8.py --wandb -s $size -m $model -p vllm
        done
    done
else
    for size in "${sizes_dev[@]}"; do
        for model in "${models_dev[@]}"; do
            echo "Running BlendSQL, Q8 with -s $size and m $model"
            python blendsql/queries/derivation/Q8/q8.py --wandb -s $size -m $model -p ollama
            echo "Evaluating BlendSQL, Q8 with -s $size and m $model executed from blendsql"
            python evaluation/derivation/Q8/eval_scripts/blendsql_q8_eval.py -s $size -m $model -p ollama
        done
    done
fi