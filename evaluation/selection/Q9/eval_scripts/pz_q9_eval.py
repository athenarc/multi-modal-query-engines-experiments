import pandas as pd

def count_true_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['nationality_gt'] == df['nationality_pred']) & (df['nationality_gt'] == "American")])

def count_false_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['nationality_gt'] != df['nationality_pred']) & (df['nationality_pred'] == "American")])

def count_true_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['nationality_gt'] != "American")])

def count_false_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['nationality_gt'] == "American")])

if __name__ == "__main__":
    df_player_labels = pd.read_csv("datasets/player_evidence_mine.csv").head(100)
    df_player_labels = df_player_labels[['Player Name', 'nationality']]

    pz_res = pd.read_csv("selection/Q9/results/pz_Q9_gemma3_12b_ollama.csv")
    pz_res = pz_res.drop(columns=['filename']).rename(columns={'contents' : 'Player Name'})
    pz_res['nationality'] = 'American'

    df = df_player_labels.merge(pz_res, on='Player Name', how='outer', suffixes=('_gt', '_pred'), indicator=True)

    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    print(f"True Positives: {tp}"
        f"\nFalse Positives: {fp}"
        f"\nTrue Negatives: {tn}"
        f"\nFalse Negatives: {fn}")

    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2f}")