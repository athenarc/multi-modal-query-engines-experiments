import pandas as pd

def count_true_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['Points_gt'] == df['Points_pred']) & (df['Points_gt'] == 17.0)])

def count_false_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['Points_gt'] != df['Points_pred']) & (df['Points_pred'] == 17.0)])

def count_true_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['Points_gt'] != 17.0)])

def count_false_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['Points_gt'] == 17.0)])

if __name__ == "__main__":

    player_labels = pd.read_csv("datasets/players_labels_100.csv")
    player_labels = player_labels[player_labels['Game ID'] < 14]
    player_labels = player_labels[['Player Name', 'Points']]

    lotus_res_default = pd.read_csv("selection/Q6/results/lotus_Q6_default_gemma3_12b_ollama.csv")
    lotus_res_default = lotus_res_default.drop(columns=["Report", "Game ID"])
    lotus_res_default['Points'] = 17.0

    df = player_labels.merge(lotus_res_default, on=["Player Name"], how="outer", suffixes=('_gt', '_pred'), indicator=True)

    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    print("--- Default Implementation ---")
    print(f"True Positives: {tp}"
        f"\nFalse Positives: {fp}"
        f"\nTrue Negatives: {tn}"
        f"\nFalse Negatives: {fn}")

    print(f"Accuracy for default implementation: {(tp + tn) / (tp + tn + fp + fn):.2f}")

    lotus_res_opt = pd.read_csv("selection/Q6/results/lotus_Q6_cascades_gemma3_12b_ollama_llama8b_vllm.csv")

    lotus_res_opt = lotus_res_opt.drop(columns=["Report", "Game ID"])
    lotus_res_opt['Points'] = 17.0

    df = player_labels.merge(lotus_res_opt, on=["Player Name"], how="outer", suffixes=('_gt', '_pred'), indicator=True)

    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    print("\n\n--- Optimized Implementation ---")
    print(f"True Positives: {tp}"
        f"\nFalse Positives: {fp}"
        f"\nTrue Negatives: {tn}"
        f"\nFalse Negatives: {fn}")
    print(f"Accuracy for optimized implementation: {(tp + tn) / (tp + tn + fp + fn):.2f}")