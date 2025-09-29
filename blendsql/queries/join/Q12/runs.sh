#!/bin/bash
sizes=(20 50 100)
models=("gemma3:12b" "llama3.1:8b")

for size in "${sizes[@]}"; do
  for model in "${models[@]}"; do
    echo "Running with -s $size and -m $model"
    python blendsql/queries/join/Q12/q12.py  --wandb -s $size -m $model -p ollama
  done
done