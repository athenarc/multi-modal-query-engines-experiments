#!/bin/bash

sizes=(1000 10000 30000)
models=("gemma3:12b") # more models to add

for size in "${sizes[@]}"; do
  for model in "${models[@]}"; do
    echo "Running with -s $size and -m $model"
    python lotus/queries/aggregation/Q14/q14.py  --wandb -s $size -m $model -p ollama
  done
done