#!/bin/bash
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for model in "${models_ollama[@]}"; do
    echo "Evaluating with -m $model"
    echo -n "Lotus "
    python evaluation/selection/Q10/eval_scripts/lotus_q10_eval.py -m $model -p ollama
    # echo -n "BlendSQL "
    # python evaluation/selection/Q10/eval_scripts/blendsql_q10_eval.py -m $model -p ollama
    echo ""
done

for model in "${models_vllm[@]}"; do
    echo "Evaluating with -m $model"
    echo -n "Lotus "
    python evaluation/selection/Q10/eval_scripts/lotus_q10_eval.py -m $model -p vllm
    # echo -n "BlendSQL "
    # python evaluation/selection/Q10/eval_scripts/blendsql_q10_eval.py -m $model -p vllm
    echo ""
done