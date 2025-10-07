import pandas as pd
from rapidfuzz import process, fuzz
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

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


    if args.provider == 'ollama':
        results_file = f"evaluation/selection/Q6/results/palimpzest_Q6_filter_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
    elif args.provider == 'vllm':
        results_file = f"evaluation/selection/Q6/results/palimpzest_Q6_filter_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

    df_player_labels = pd.read_csv("datasets/rotowire/player_labels.csv")
    df_player_labels = df_player_labels[df_player_labels['Game ID'] < args.size]
    df_player_labels = df_player_labels[['Game ID', 'Player Name', 'Points']]

    pz_res = pd.read_csv(results_file)

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
