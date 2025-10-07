import pandas as pd
from rapidfuzz import process, fuzz

player_evi = pd.read_csv("player_evidence_mine.csv")[['Player Name', 'birth_date']].dropna(subset=['birth_date'])
blendsql_evi = pd.read_csv("projection/Q4/blendsql_q4_llama3_3_70b_ollama.csv")
blendsql_evi.rename(columns={'player_name': 'Player Name', '_col_1' : 'birth_date'}, inplace=True)

print(blendsql_evi)
blendsql_evi['birth_date'] = blendsql_evi['birth_date'].str.replace('/', '.')

df = player_evi.merge(blendsql_evi, left_on='Player Name', right_on='Player Name', how='outer')

df["match"] = df.apply(
    lambda row: (row["birth_date_y"] in row["birth_date_x"]) or 
                (row["birth_date_x"] in row["birth_date_y"]),
    axis=1
)

print(f"Accuracy: {df['match'].mean():.2%}")