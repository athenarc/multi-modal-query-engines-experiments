#!/bin/bash

# Usage: ./aggregation_runs.sh <provider> <query_numbers...>
# Example: ./aggregation_runs.sh vllm 16 17

provider="$1"
shift 

# Validate provider
if [[ -z "$provider" || ("$provider" != "ollama" && "$provider" != "vllm")]]; then
    echo "Error: provider argument required (ollama | vllm)"
    echo "Usage: $0 <provider> <query_numbers...>"
    exit 1
fi

# Validate query list
queries=("$@")
if [[ ${#queries[@]} -eq 0 ]]; then # Run them all by default
    queries=(16 17 18)
fi

for q in "${queries[@]}"; do
    if [[ "$q" != "16" && "$q" != "17" && "$q" != "18" ]]; then
        echo "Error: Unsupported query number '$q'. Supported queries are 16, 17, 18."
        exit 1
    fi
done

# Experiments
sizes=(10 20 30 40 50 60)
models_ollama=("gemma3:12b-128k")
models_vllm=("RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8")

if [[ "$provider" == "ollama" ]]; then
    models=("${models_ollama[@]}")
else
    models=("${models_vllm[@]}")
fi

for qnum in "${queries[@]}"; do
    Q_dir="Q${qnum}"
    query="q${qnum}"

    for size in "${sizes[@]}"; do
        for model in "${models[@]}"; do
            echo "----------------------------------------------------------"
            echo "Running Lotus $Q_dir: size=$size, model=$model, provider=$provider"
            echo "----------------------------------------------------------"
            
            python "lotus/queries/aggregation/$Q_dir/${query}.py" --wandb -s "$size" -m "$model" -p "$provider"
            python "evaluation/aggregation/$Q_dir/eval_scripts/${query}_eval.py" --size "$size"
        done
    done
done