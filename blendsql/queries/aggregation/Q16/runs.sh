#!/bin/bash

models=("gemma3:12b" "llama3.1:8b")

for model in "${models[@]}"; do
    echo "Running with -s $size and -m $model"
    python blendsql/queries/aggregation/Q16/q16.py  --wandb -m $model -p ollama
done