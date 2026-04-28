#!/bin/bash

sizes=(100 400 728)

for size in "${sizes[@]}"; do
    echo "Evaluating with -s $size and -m $model"
    python evaluation/aggregation/Q18/eval_scripts/q18_eval.py -s $size
done