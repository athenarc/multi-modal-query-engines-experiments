import pandas as pd
from rapidfuzz import process, fuzz

player_evi = pd.read_csv("player_evidence_mine.csv")[['Player Name', 'birth_date']].dropna(subset=['birth_date'])
lotus_evi = pd.read_csv("projection/Q4/lotus_q4_llama3_3_70b_ollama.csv")
lotus_evi.rename(columns={'birthdate' : 'birth_date'}, inplace=True)
lotus_evi['birth_date'] = lotus_evi['birth_date'].str.replace('/', '.')

df = player_evi.merge(lotus_evi, left_on='Player Name', right_on='Player Name', how='outer')

df["match"] = df.apply(
    lambda row: (row["birth_date_y"] in row["birth_date_x"]) or 
                (row["birth_date_x"] in row["birth_date_y"]),
    axis=1
)

df.to_csv('fuzzmatch_q4_lotus_eval.csv', index=False)

print(f"Accuracy: {df['match'].mean():.2%}")