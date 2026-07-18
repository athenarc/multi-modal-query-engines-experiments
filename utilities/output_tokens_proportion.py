"""
Aggregate query benchmark results by system, class_name, and task_name,
then compute the proportion of output tokens out of total tokens.

Usage:
    python aggregate_output_tokens.py path/to/data.csv
"""

import sys
import pandas as pd


def aggregate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df['model_name'] == "RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8"]
    # df = df[df['model_name'] == "Qwen/Qwen3-8B"]


    agg = (
        df.groupby(["system", "class_name", "task_name"])
        .agg(
            output_tokens=("output_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            n_rows=("output_tokens", "count"),
        )
        .reset_index()
    )

    agg["output_pct"] = (agg["output_tokens"] / agg["total_tokens"] * 100).round(2)

    agg = agg.sort_values(["class_name", "task_name", "output_pct"], ascending=[True, True, False])

    return agg



if __name__ == "__main__":
    result = aggregate("results/stats/scalability/stats_all.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    print(result.to_string(index=False))

    out_path = "aggregated_output_token_proportions.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved full results to {out_path}")