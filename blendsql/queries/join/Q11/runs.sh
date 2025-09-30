#!/bin/bash
sizes=(20 50 100)
models=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
  for model in "${models[@]}"; do
    echo "Running with -s $size and -m $model"
    python blendsql/queries/join/Q11/q11.py  --wandb -s $size -m $model -p vllm
  done
done