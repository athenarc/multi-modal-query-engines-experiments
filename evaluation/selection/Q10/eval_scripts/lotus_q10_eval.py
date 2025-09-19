import pandas as pd
import os

def count_true_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['founded_gt'] < 1970)])

def count_false_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['founded_gt'] >= 1970)])

def count_true_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['founded_gt'] >= 1970)])

def count_false_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['founded_gt'] < 1970)])

if __name__ == "__main__":
    team_evidence = pd.read_csv("datasets/team_evidence.csv")
    team_evidence = team_evidence[['Team Name', 'founded']]

    lotus_res_default = pd.read_csv("selection/Q10/results/lotus_q10_filter_default_gemma3_12b_ollama.csv")
    lotus_res_default = lotus_res_default.rename(columns={'team_name' : "Team Name"})
    lotus_res_default['founded'] = 1900 # Just a random value < 1970

    df = team_evidence.merge(lotus_res_default, on=['Team Name'], how='outer', suffixes=('_gt', '_pred'), indicator=True)

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

    lotus_res_opt = pd.read_csv("selection/Q10/results/lotus_q10_filter_cascades_gemma3_12b_ollama_llama8b_vllm.csv")
    lotus_res_opt = lotus_res_opt.rename(columns={'team_name' : "Team Name"})
    lotus_res_opt['founded'] = 1900 # Just a random value < 1970

    df = team_evidence.merge(lotus_res_opt, on=['Team Name'], how='outer', suffixes=('_gt', '_pred'), indicator=True)

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
