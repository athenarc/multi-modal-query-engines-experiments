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

sizes=(10 20 30 40 50)
# sizes=(10 20)

models_ollama=("gemma3:12b")
models_vllm=("RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8")
# models_vllm=("meta-llama/Llama-3.1-8B-Instruct")


# Select models based on provider
if [[ "$provider" == "ollama" ]]; then
    models=("${models_ollama[@]}")
else
    models=("${models_vllm[@]}")
fi

# Run default.py
for size in "${sizes[@]}"; do
    for model in "${models[@]}"; do
        echo "Running Lotus-default, Q15 with -s $size and -m $model"
        python lotus/queries/join/Q15/default.py --wandb -s $size -m $model -p $provider
        echo "Evaluating Lotus-default, Q15 with -s $size and m $model executed from Lotus"
        python evaluation/join/Q15/eval_scripts/q15_eval.py --system lotus -s $size -m $model -p $provider
    done
done
