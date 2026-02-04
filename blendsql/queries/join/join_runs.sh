#!/bin/bash

# Usage: ./run_queries.sh <provider> <query_numbers...>
# Example: ./run_queries.sh ollama 13 15

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
    queries=(13 14 15)
fi

for q in "${queries[@]}"; do
    if [[ "$q" != "13" && "$q" != "14" && "$q" != "15" ]]; then
        echo "Error: Unsupported query number '$q'. Supported queries are 13, 14, 15."
        exit 1
    fi
done

# Experiments
sizes=(10 20 30 40 50)
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
            echo "Running BlendSQL $Q_dir: size=$size, model=$model, provider=$provider"
            echo "----------------------------------------------------------"
            
            python "blendsql/queries/join/$Q_dir/${query}.py" --wandb -s "$size" -m "$model" -p "$provider"
            python "evaluation/join/$Q_dir/eval_scripts/${query}_eval.py" --size "$size"
        done
    done
done