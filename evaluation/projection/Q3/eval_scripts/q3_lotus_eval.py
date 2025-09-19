import pandas as pd
from rapidfuzz import process, fuzz

player_evi = pd.read_csv("datasets/player_evidence_mine.csv")[['Player Name', 'nationality']].dropna(subset=['nationality'])
lotus_evi = pd.read_csv("projection/Q3/results/lotus_q3_gemma3_12b_ollama.csv")

df = player_evi.merge(lotus_evi, left_on='Player Name', right_on='Player Name', how='outer')
df['nationality_y'] = df['nationality_y'].str.replace('\n', '')

df["match"] = df.apply(
    lambda row: (row["nationality_y"] in row["nationality_x"]) or 
                (row["nationality_x"] in row["nationality_y"]) or
                (fuzz.ratio(row['nationality_x'], row['nationality_y']) >= 70),
    axis=1
)

print(f"Accuracy: {df['match'].mean():.2%}")