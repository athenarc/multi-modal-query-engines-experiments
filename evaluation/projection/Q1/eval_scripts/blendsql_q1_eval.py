import pandas as pd
from rapidfuzz import process, fuzz
import argparse

def match_name(name, choices, scorer=fuzz.ratio, threshold=60):
    if not choices:
        return None
    match = process.extractOne(name, choices, scorer=scorer, score_cutoff=threshold)
    return match[0] if match else None

def match_group(group):
    game_id = group.name
    choices = df_labels.loc[df_labels['Game ID'] == game_id, 'Player Name'].tolist()
    group['matched_player'] = group['player_name'].apply(lambda x: match_name(x, choices))
    return group

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

df_labels = pd.read_csv('datasets/players_labels.csv')
df_labels = df_labels[df_labels['Game ID'] < args.size]

if args.provider == 'ollama':
    results_file = f"evaluation/projection/Q1/results/blendsql_Q1_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    results_file = f"evaluation/projection/Q1/results/blendsql_Q1_map_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

df_blendsql = pd.read_csv(results_file)

df_blendsql.rename(columns={"Game_ID": "Game ID", "points": "Points", "assists": "Assists", "total_rebounds": "Total rebounds", "steals": "Steals", "blocks": "Blocks"}, inplace=True) 
df_blendsql = df_blendsql.groupby('Game ID', group_keys=False).apply(match_group)

df = df_labels.merge(df_blendsql, left_on=['Game ID', 'Player Name'], right_on=['Game ID', 'matched_player'], how='left', indicator=True)

df.drop(columns=["Defensive rebounds", "Offensive rebounds", "3-pointers attempted", "3-pointers made", "Field goals attempted", "Field goals made", "Free throws attempted", "Free throws made", "Minutes played", "Personal fouls", "Turnovers", "Field goal percentage", "Free throw percentage", "3-pointer percentage"], inplace=True)

cols = ["Points", "Assists", "Total rebounds", "Blocks", "Steals"]

for col in cols:
    xcol, ycol = f"{col}_x", f"{col}_y"
    df[f"{col}_match"] = (df[xcol].fillna(-1) == df[ycol].fillna(-1))

for col in cols:
    acc = df[f"{col}_match"].mean()
    print(f"{col} accuracy: {acc:.2%}")

total_accuracy = df[[f"{col}_match" for col in cols]].stack().mean()
print(f"Total accuracy: {total_accuracy:.2%}")

df_gtrue = df_labels[['Game ID', 'Player Name']]
df_pred = df[['Game ID', 'matched_player']].rename(columns={'matched_player': 'Player Name'})

merged = df_gtrue.merge(df_pred, on=['Game ID', 'Player Name'], how='outer', indicator=True)

TP = len(merged[merged['_merge'] == 'both'])
FP = len(merged[merged['_merge'] == 'right_only'])
FN = len(merged[merged['_merge'] == 'left_only'])

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"\nF1-score: {f1:.2f}\n")