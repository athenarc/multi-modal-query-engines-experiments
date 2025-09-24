import pandas as pd

def match_team_names(team_name):
    choices = players_evi['Team Name']

    for choice in choices:
        if team_name.lower() in choice.lower() \
        or choice.lower() in team_name.lower():
            return choice
        
    return team_name

players_evi = pd.read_csv("datasets/rotowire/player_evidence_mine.csv").head(50)[['Player Name', 'team_2015']].rename(columns={'team_2015' : "Team Name"})
players_evi["Team Name"] = players_evi["Team Name"].str.split(",")
players_evi = players_evi.explode("Team Name")
players_evi['Team Name'] = players_evi['Team Name'].str.strip()
players_evi.fillna('None', inplace=True)
blendsql_res = pd.read_csv("evaluation/join/Q11/results/blendsql_q11_updated.csv")[['player_name', 'team_name']].rename(columns={'player_name': 'Player Name', 'team_name' : 'Team Name'})

blendsql_res['Team Name'] = blendsql_res['Team Name'].apply(match_team_names)

df = players_evi.merge(blendsql_res, on=['Player Name', 'Team Name'], how='outer', indicator=True)

accuracy = len(df[df['_merge'] == 'both']) / blendsql_res.shape[0]

print(f"Accuracy : {accuracy:.2f}")