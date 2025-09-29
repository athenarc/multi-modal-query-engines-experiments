#!/bin/bash

models=("gemma3:12b") # more models to add

for model in "${models[@]}"; do
    echo "Running with -s $size and -m $model"
    python lotus/queries/aggregation/Q16/q16.py  --wandb -s $size -m $model -p ollama
done