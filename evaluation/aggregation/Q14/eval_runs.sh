#!/bin/bash

sizes=(1000 10000 30000)

for size in "${sizes[@]}"; do
    echo "Evaluating with -s $size and -m $model"
    python evaluation/aggregation/Q14/eval_scripts/q14_eval.py -s $size
done