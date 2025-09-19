import pandas as pd
from rapidfuzz import process, fuzz

df_labels = pd.read_csv('datasets/team_labels_100.csv')
pz_labels = pd.read_csv("projection/Q2/results/pz_q2_gemma3_12b.csv")
pz_labels['Game ID'] = pz_labels['filename'].str.extract(r'report_(\d+)\.txt').astype(int)
pz_labels.sort_values(['Game ID'], inplace=True)
pz_labels['Game ID'] = pz_labels['Game ID'] - 1

print(pz_labels)

def match_name(name, choices, scorer=fuzz.ratio, threshold=30):
    if not choices:
        return None
    match = process.extractOne(name, choices, scorer=scorer, score_cutoff=threshold)
    return match[0] if match else None

def match_group(group):
    game_id = group.name
    choices = df_labels.loc[df_labels['Game ID'] == game_id, 'Team Name'].tolist()
    group['matched_team'] = group['team_name'].apply(lambda x: match_name(x, choices))
    return group

pz_labels = pz_labels.groupby('Game ID', group_keys=False).apply(match_group)
pz_labels.rename(columns={'Total Points': 'Total points'}, inplace=True)

print(pz_labels)


df = df_labels.merge(pz_labels, left_on=['Game ID', 'Team Name'], right_on=['Game ID', 'matched_team'], how='left', indicator=True)

df.drop(columns=["Points in 4th quarter", "Percentage of field goals", "Rebounds", "Number of team assists", "Points in 3rd quarter", "Turnovers", "Percentage of 3 points", "Points in 1st quarter", "Points in 2nd quarter"], inplace=True)
cols = ["Wins", "Losses", "Total points"]

print(df)

for col in cols:
    xcol, ycol = f"{col}_x", f"{col}_y"
    df[f"{col}_match"] = (df[xcol].fillna(-1) == df[ycol].fillna(-1))

for col in cols:
    acc = df[f"{col}_match"].mean()
    print(f"{col} accuracy: {acc:.2%}")

total_accuracy = df[[f"{col}_match" for col in cols]].stack().mean()
print(f"Total accuracy: {total_accuracy:.2%}")

df_gtrue = df_labels[['Game ID', 'Team Name']]
df_pred = df[['Game ID', 'matched_team']].rename(columns={'matched_team': 'Team Name'})

merged = df_gtrue.merge(df_pred, on=['Game ID', 'Team Name'], how='outer', indicator=True)

TP = len(merged[merged['_merge'] == 'both'])
FP = len(merged[merged['_merge'] == 'right_only'])
FN = len(merged[merged['_merge'] == 'left_only'])

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"\nF1-score: {f1:.2f}\n")