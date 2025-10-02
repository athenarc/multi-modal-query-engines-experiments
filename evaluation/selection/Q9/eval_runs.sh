#!/bin/bash
sizes=(100 250 500)
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Evaluating with -s $size and -m $model"
        echo -n "Lotus "
        python evaluation/selection/Q9/eval_scripts/lotus_q9_eval.py -s $size -m $model -p ollama
        echo -n "Palimpzest "
        python evaluation/selection/Q9/eval_scripts/pz_q9_eval.py -s $size -m $model -p ollama
        echo ""
    done

    for model in "${models_vllm[@]}"; do
        echo "Evaluating with -s $size and -m $model"
        echo -n "Lotus "
        python evaluation/selection/Q9/eval_scripts/lotus_q9_eval.py -s $size -m $model -p vllm
        echo -n "Palimpzest "
        python evaluation/selection/Q9/eval_scripts/pz_q9_eval.py -s $size -m $model -p vllm
        echo ""
    done
done