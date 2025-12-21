#!/bin/bash

provider="$1"

if [[ -z "$provider" ]]; then
    echo "Error: provider argument required (ollama | vllm)"
    exit 1
fi

if [[ "$provider" != "ollama" && "$provider" != "vllm" ]]; then
    echo "Error: provider must be 'ollama' or 'vllm'"
    exit 1
fi

sizes=(500 1000 2000 4000)

models_ollama=("llama3.3:70b")
models_vllm=("RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8")

# Select models based on provider
if [[ "$provider" == "ollama" ]]; then
    models=("${models_ollama[@]}")
else
    models=("${models_vllm[@]}")
fi

# Run default.py
for size in "${sizes[@]}"; do
    for model in "${models[@]}"; do
        echo "Running Q11 default with -s $size and -m $model"
        python palimpzest/queries/selection/Q11/q11.py --wandb -s $size -m $model -p $provider
        echo "Evaluating Palimpzest, Q11 with -s $size and -m $model"
        python evaluation/selection/Q11/eval_scripts/pz_q11_eval.py -s $size -m $model -p $provider
    done
done