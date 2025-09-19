import pandas as pd
from rapidfuzz import process, fuzz

def match_name(name, choices, scorer=fuzz.ratio, threshold=60):
    if not choices:
        return None
    match = process.extractOne(name, choices, scorer=scorer, score_cutoff=threshold)
    return match[0] if match else None

def match_group(group):
    game_id = group.name
    choices = df_player_labels.loc[df_player_labels['Game ID'] == game_id, 'Player Name'].tolist()
    group['matched_player'] = group['Player Name'].apply(lambda x: match_name(x, choices))
    return group

def count_true_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['Points_gt'] == df['Points_pred']) & (df['Points_gt'] == 17.0)])

def count_false_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['Points_gt'] != df['Points_pred']) & (df['Points_pred'] == 17.0)])

def count_true_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['Points_gt'] != 17.0)])

def count_false_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['Points_gt'] == 17.0)])

if __name__ == "__main__":
    df_player_labels = pd.read_csv("datasets/players_labels_100.csv")
    df_player_labels = df_player_labels[df_player_labels['Game ID'] < 14]
    df_player_labels = df_player_labels[['Game ID', 'Player Name', 'Points']]

    pz_res = pd.read_csv("selection/Q6/results/pz_Q6_gemma3_12b_ollama.csv")
    pz_res['Game ID'] = pz_res['filename'].str.extract(r'report_(\d+)\.txt').astype(int)
    pz_res.sort_values(by=['Game ID'], inplace=True)
    pz_res['Game ID'] = pz_res['Game ID'] - 1
    pz_res['Points'] = 17.0
    pz_res.drop(columns=['filename', 'contents'], inplace=True)
    pz_res.rename(columns={'player_name': 'Player Name'}, inplace=True)

    pz_res = pz_res.groupby('Game ID', group_keys=False).apply(match_group)
    pz_res = pz_res.drop(columns=['Player Name']).rename(columns={'matched_player': 'Player Name'})

    df = df_player_labels.merge(pz_res, on=["Player Name"], how="outer", suffixes=('_gt', '_pred'), indicator=True)

    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    print(f"True Positives: {tp}"
        f"\nFalse Positives: {fp}"
        f"\nTrue Negatives: {tn}"
        f"\nFalse Negatives: {fn}")

    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2f}")
