#!/bin/bash

# Usage: ./selection_runs.sh <provider> <query_numbers...>
# Example: ./selection_runs.sh vllm 9 10 11

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
    queries=(9 10 11 12)
fi

for q in "${queries[@]}"; do
    if [[ "$q" != "9" && "$q" != "10" && "$q" != "11" && "$q" != "12" ]]; then
        echo "Error: Unsupported query number '$q'. Supported queries are 9, 10, 11, 12."
        exit 1
    fi
done

# Experiments
sizes=(500 1000 2000 4000)
models_ollama=("gemma3:12b")
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
            echo "Running Palimpzest $Q_dir: size=$size, model=$model, provider=$provider"
            echo "----------------------------------------------------------"
            
            python "palimpzest/queries/selection/$Q_dir/${query}.py" --wandb -s "$size" -m "$model" -p "$provider"
            python "evaluation/selection/$Q_dir/eval_scripts/pz_${query}_eval.py" --size "$size" -m $model -p $provider
        done
    done
done