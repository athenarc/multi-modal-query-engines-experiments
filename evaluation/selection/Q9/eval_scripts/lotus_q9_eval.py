import pandas as pd
import os

def count_true_positives(df):
    if (df['_merge'] == 'both').any():
        return len(df[(df['_merge'] == 'both') & df.apply(lambda row: pd.notna(row['nationality_pred']) and pd.notna(row['nationality_gt']) and row['nationality_pred'] in row['nationality_gt'], axis=1)])
    return 0

def count_false_positives(df):
    if (df['_merge'] == 'both').any():
        return len(df[(df['_merge'] == 'both') & df.apply(lambda row: pd.notna(row['nationality_pred']) and pd.notna(row['nationality_gt']) and row['nationality_pred'] not in row['nationality_gt'] and row['nationality_pred'] == "American", axis=1)])
    return 0

def count_true_negatives(df):
    if (df['_merge'] == 'left_only').any():
        return len(df[(df['_merge'] == 'left_only') & (~df['nationality_gt'].str.contains("American", na=False))])
    return len(df[(df['nationality'] != 'American')])

def count_false_negatives(df):
    if (df['_merge'] == 'left_only').any():
        return len(df[(df['_merge'] == 'left_only') & (df['nationality_gt'].str.contains("American", na=False))])
    return len(df[df['nationality'] == 'American'])

if __name__ == "__main__":
    player_evidence = pd.read_csv("datasets/player_evidence_mine.csv").head(100)
    player_evidence = player_evidence[['Player Name', 'nationality']]

    if os.stat("selection/Q9/results/lotus_q9_filter_default_gemma3_12b_ollama.csv").st_size == 0:
        lotus_res_default = pd.DataFrame(columns=['Player Name', 'nationality'])
    else:
        lotus_res_default = pd.read_csv("selection/Q9/results/lotus_q9_filter_default_gemma3_12b_ollama.csv")
        lotus_res_default = lotus_res_default.rename(columns={'player_name' : 'Player Name'})
        lotus_res_default['nationality'] = 'American'

    df = player_evidence.merge(lotus_res_default, on=['Player Name'], how='outer', suffixes=('_gt', '_pred'), indicator=True)

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

    if os.stat("selection/Q9/results/lotus_q9_filter_cascades_gemma3_12b_ollama_llama8b_vllm.csv").st_size == 0:
        lotus_res_opt = pd.DataFrame(columns=['Player Name', 'nationality'])
    else:
        lotus_res_opt = pd.read_csv("selection/Q9/results/lotus_q9_filter_cascades_gemma3_12b_ollama_llama8b_vllm.csv")
        lotus_res_opt = lotus_res_opt.rename(columns={'player_name' : 'Player Name'})
        lotus_res_opt['nationality'] = 'American'

    df = player_evidence.merge(lotus_res_opt, on=['Player Name'], how='outer', suffixes=('_gt', '_pred'), indicator=True)
    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    print("--- Optimized Implementation ---")
    print(f"True Positives: {tp}"
        f"\nFalse Positives: {fp}"
        f"\nTrue Negatives: {tn}"
        f"\nFalse Negatives: {fn}")

    print(f"Accuracy for optimized implementation: {(tp + tn) / (tp + tn + fp + fn):.2f}")
