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

models_ollama=("gemma3:12b")
models_vllm=("RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8")

# Select models based on provider
if [[ "$provider" == "ollama" ]]; then
    models=("${models_ollama[@]}")
else
    models=("${models_vllm[@]}")
fi

for size in "${sizes[@]}"; do
    for model in "${models[@]}"; do
        echo "Running Palimpzest, Q10 with -s $size and -m $model"
        python palimpzest/queries/selection/Q10/q10.py --wandb -s $size -m $model -p $provider
        echo "Evaluating Palimpzest, Q10 with -s $size and -m $model"
        python evaluation/selection/Q10/eval_scripts/pz_q10_eval.py -s $size -m $model -p $provider
    done
done