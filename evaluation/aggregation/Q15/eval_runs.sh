#!/bin/bash

sizes=(100 400 728)

for size in "${sizes[@]}"; do
    echo "Evaluating with -s $size and -m $model"
    python evaluation/aggregation/Q15/eval_scripts/q15_eval.py -s $size
done