#!/bin/bash

# Usage: ./run_queries.sh <provider> <query_numbers...>
# Example: ./run_queries.sh ollama 1 3 4

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
    queries=(1 2 3 4 5 6 7 8)
fi

for q in "${queries[@]}"; do
    if [[ "$q" != "1" && "$q" != "2" && "$q" != "3" && "$q" != "4" && "$q" != "5" && "$q" != "6" && "$q" != "7" && "$q" != "8" ]]; then
        echo "Error: Unsupported query number '$q'. Supported queries are 1, 2, 3, 4, 5, 6, 7, 8."
        exit 1
    fi
done

# Experiments
sizes_dev=(50 100 200 300 500)
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
            
            python "blendsql/queries/aggregation/$Q_dir/${query}.py" --wandb -s "$size" -m "$model" -p "$provider"
            python "evaluation/aggregation/$Q_dir/eval_scripts/${query}_eval.py" --size "$size"
        done
    done
done