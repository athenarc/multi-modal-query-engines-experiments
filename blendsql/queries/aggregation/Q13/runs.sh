#!/bin/bash
sizes=(1000 10000 50000)
models=("gemma3:12b" "llama3.1:8b")

for size in "${sizes[@]}"; do
  for model in "${models[@]}"; do
    echo "Running with -s $size and -m $model"
    python blendsql/queries/aggregation/Q13/q13.py  --wandb -s $size -m $model -p ollama
  done
done