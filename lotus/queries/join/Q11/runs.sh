#!/bin/bash
sizes=(20 50 100)
models=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
  for model in "${models[@]}"; do
    echo "Running with -s $size and -m $model"
    python lotus/queries/join/Q11/default.py  --wandb -s $size -m $model -p vllm
  done
done