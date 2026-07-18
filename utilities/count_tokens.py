import pandas as pd
from transformers import AutoTokenizer
from typing import Union, List, Dict

def analyze_token_counts(
    tokenizer,
    csv_path: str, 
    columns: Union[str, List[str]], 
) -> Dict[str, Dict[str, float]]:
    if isinstance(columns, str):
        columns = [columns]

    print(f"\nLoading data from: '{csv_path}'...")
    df = pd.read_csv(csv_path).dropna(subset=columns)
    
    results = {}

    for col in columns:
        if col not in df.columns:
            print(f"\n[Warning] Column '{col}' not found in the CSV. Skipping.")
            continue

        texts = df[col].fillna("").astype(str).tolist()

        encodings = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_tensors=None
        )

        token_counts = [len(ids) for ids in encodings["input_ids"]]

        if not token_counts:
            continue

        avg_tokens = sum(token_counts) / len(token_counts)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)

        results[col] = {
            "rows": len(token_counts),
            "average": avg_tokens,
            "min": min_tokens,
            "max": max_tokens
        }

        print(f"\n=== Token Statistics for '{col}' ===")
        print(f"Rows: {len(token_counts)}")
        print(f"Average tokens: {avg_tokens:.2f}")
        print(f"Min tokens: {min_tokens}")
        print(f"Max tokens: {max_tokens}")

    return results

if __name__ == "__main__":
    model_name: str = "meta-llama/Meta-Llama-3-70B-Instruct"
    print(f"Loading tokenizer: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Token statistics for Sportsett summaries
    stats_summaries = analyze_token_counts(
        tokenizer=tokenizer,
        csv_path="datasets/nba/summaries.csv",
        columns="summary"
    )

    # Token statistics for Rotten Tomatoes textual columns
    stats_movies = analyze_token_counts(
        tokenizer=tokenizer,
        csv_path="datasets/rotten_tomatoes/movies.csv",
        columns=["movie_info", "critics_consensus"]
    )

    stats_reviews = analyze_token_counts(
        tokenizer=tokenizer,
        csv_path="datasets/rotten_tomatoes/reviews.csv",
        columns="review_content"
    )