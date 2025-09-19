import pandas as pd

def count_true_positives(df):
    true_positives = df[(df['sentiment_gt'] == 'positive') & (df['sentiment_pred'] == 'positive')]
    return len(true_positives)

def count_false_positives(df):
    false_positives = df[(df['sentiment_gt'] == 'negative') & (df['sentiment_pred'] == 'positive')]
    return len(false_positives)

def count_true_negatives(df):
    true_negatives = df[(df['_merge'] == 'left_only') & (df['sentiment_gt'] == 'negative')]
    return len(true_negatives)

def count_false_negatives(df):
    false_negatives = df[(df['_merge'] == 'left_only') & (df['sentiment_gt'] == 'positive')]
    return len(false_negatives)


if __name__ == "__main__":
    imdb_dataset = pd.read_csv("datasets/imdb_dataset.csv").head(100)
    pz_results = pd.read_csv("selection/Q5/results/pz_Q5_gemma3_12b_ollama.csv")

    pz_results["sentiment"] = "positive"
    pz_results = pz_results.drop(columns=["filename"]).rename(columns={"contents": "review"})

    df = imdb_dataset.merge(pz_results, on="review", how="outer", suffixes=('_gt', '_pred'), indicator=True)

    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    print(f"True Positives: {tp}"
        f"\nFalse Positives: {fp}"
        f"\nTrue Negatives: {tn}"
        f"\nFalse Negatives: {fn}")

    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2f}")