#!/bin/bash
sizes=(7 14 29)
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")

for size in "${sizes[@]}"; do
    for model in "${models_ollama[@]}"; do
        echo "Evaluating with -s $size and -m $model"
        echo -n "Lotus "
        python evaluation/selection/Q6/eval_scripts/lotus_q6_eval.py -s $size -m $model -p ollama
        # echo -n "BlendSQL "
        # python evaluation/selection/Q6/eval_scripts/blendsql_Q6_eval.py -s $size -m $model -p ollama
        echo ""
    done

    for model in "${models_vllm[@]}"; do
        echo "Evaluating with -s $size and -m $model"
        echo -n "Lotus "
        python evaluation/selection/Q6/eval_scripts/lotus_q6_eval.py -s $size -m $model -p vllm
        # echo -n "BlendSQL "
        # python evaluation/selection/Q6/eval_scripts/blendsql_Q6_eval.py -s $size -m $model -p vllm
        echo ""
    done
done