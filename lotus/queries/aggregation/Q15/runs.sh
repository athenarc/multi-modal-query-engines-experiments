#!/bin/bash

sizes=(100 400 728)
models=("gemma3:12b") # more models to add

for size in "${sizes[@]}"; do
  for model in "${models[@]}"; do
    echo "Running with -s $size and -m $model"
    python lotus/queries/aggregation/Q15/q15.py  --wandb -s $size -m $model -p ollama
  done
done