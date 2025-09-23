import pandas as pd

def match_team_names(team_name):
    choices = players_evi['Team Name']

    for choice in choices:
        if team_name.lower() in choice.lower() \
        or choice.lower() in team_name.lower():
            return choice
        
    return team_name

players_evi = pd.read_csv("datasets/player_evidence_mine.csv").head(50)[['Player Name', 'team_2015']].rename(columns={'team_2015' : "Team Name"})
players_evi["Team Name"] = players_evi["Team Name"].str.split(",")
players_evi = players_evi.explode("Team Name")
players_evi['Team Name'] = players_evi['Team Name'].str.strip()
players_evi.fillna('None', inplace=True)

lotus_res = pd.read_csv("join/Q11/results/lotus_Q11_join_default_gemma3_12b_ollama.csv")

lotus_res['Team Name'] = lotus_res['Team Name'].apply(match_team_names)

df = players_evi.merge(lotus_res, on=['Player Name', 'Team Name'], how='outer', indicator=True)

accuracy = len(df[df['_merge'] == 'both']) / lotus_res.shape[0]

print(f"Accuracy : {accuracy:.2f}")