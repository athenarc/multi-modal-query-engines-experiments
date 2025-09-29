#!/bin/bash
sizes=(10 20 40)
models=("gemma3:12b" "llama3.1:8b")

for size in "${sizes[@]}"; do
  for model in "${models[@]}"; do
    echo "Running with -s $size and -m $model"
    python lotus/queries/join/Q12/default.py  --wandb -s $size -m $model -p ollama
  done
done