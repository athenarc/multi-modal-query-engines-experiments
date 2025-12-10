#!/bin/bash
# Runs for Q4
#!/bin/bash
sizes=(10 20 50)
models_ollama=("gemma3:12b" "llama3.3:70b")
models_vllm=("meta-llama/Llama-3.1-8B-Instruct")
models_dev=("RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8")
# models_dev=("Qwen/Qwen3-8B")
sizes_dev=(50 100 200 300 500)

dev="${1:-}"


if [ "$dev" != "dev" ]; then
    for size in "${sizes[@]}"; do
        for model in "${models_ollama[@]}"; do
            echo "Running Q4 with -s $size and m $model"
            python blendsql/queries/derivation/Q4/q4.py --wandb -s $size -m $model -p ollama
        done
    done

    for size in "${sizes[@]}"; do
        for model in "${models_vllm[@]}"; do
            echo "Running Q4 with -s $size and m $model"
            python blendsql/queries/derivation/Q4/q4.py --wandb -s $size -m $model -p vllm
        done
    done
else
    for size in "${sizes_dev[@]}"; do
        for model in "${models_dev[@]}"; do
            echo "Running Q4 with -s $size and m $model"
            python blendsql/queries/derivation/Q4/q4.py  -s $size -m $model -p vllm
            echo "Evaluating Q4 with -s $size and m $model executed from blendsql"
            python evaluation/derivation/Q4/eval_scripts/blendsql_q4_eval.py -s $size -m $model -p vllm
        done
    done
fi