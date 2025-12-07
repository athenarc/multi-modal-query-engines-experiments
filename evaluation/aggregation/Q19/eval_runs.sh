#!/bin/bash

sizes=(100 400 728)

for size in "${sizes[@]}"; do
    echo "Evaluating with -s $size and -m $model"
    python evaluation/aggregation/Q19/eval_scripts/q19_eval.py -s $size
done