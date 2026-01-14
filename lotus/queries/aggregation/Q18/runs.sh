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

sizes=(10 20 30 40 50 60)

models_ollama=("gemma3:12b-128k")
# models_vllm=("meta-llama/Llama-3.1-8B-Instruct")
models_vllm=("RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8")


# Select models based on provider
if [[ "$provider" == "ollama" ]]; then
    models=("${models_ollama[@]}")
else
    models=("${models_vllm[@]}")
fi


for size in "${sizes[@]}"; do
    for model in "${models[@]}"; do
        echo "Running Lotus, Q18 with -s $size and -m $model"
        python lotus/queries/aggregation/Q18/q18.py --wandb -s $size -m $model -p $provider
        python evaluation/aggregation/Q18/eval_scripts/q18_eval.py --size $size
    done
done
