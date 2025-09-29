import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=50, const=50, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

def match_team_names(team_name):
    choices = players_evi['Team Name']

    for choice in choices:
        if team_name.lower() in choice.lower() \
        or choice.lower() in team_name.lower():
            return choice
        
    return team_name

players_evi = pd.read_csv("datasets/rotowire/player_evidence_mine.csv").head(args.size)[['Player Name', 'team_2015']].rename(columns={'team_2015' : "Team Name"})
players_evi["Team Name"] = players_evi["Team Name"].str.split(",")
players_evi = players_evi.explode("Team Name")
players_evi['Team Name'] = players_evi['Team Name'].str.strip()
players_evi.fillna('None', inplace=True)

results_file = f"evaluation/join/Q11/results/lotus_Q11_join_default_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
lotus_res = pd.read_csv(results_file)

lotus_res['Team Name'] = lotus_res['Team Name'].apply(match_team_names)

df = players_evi.merge(lotus_res, on=['Player Name', 'Team Name'], how='outer', indicator=True)

accuracy = len(df[df['_merge'] == 'both']) / lotus_res.shape[0]

print(f"Accuracy : {accuracy:.2f}")