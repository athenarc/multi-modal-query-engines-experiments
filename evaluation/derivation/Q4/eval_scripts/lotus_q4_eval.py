import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
parser.add_argument("-e", "--extract", action='store_true', help="Evaluate extract instead of map")
args = parser.parse_args()

df_reports = pd.read_csv('datasets/rotowire/reports_table.csv').rename(columns={'Game_ID': 'Game ID'})
df_player_names = pd.read_csv('datasets/rotowire/player_labels.csv')[['Player Name', 'Game ID', "Assists", "Total rebounds", "Blocks", "Steals"]].head(args.size)
df_labels = pd.merge(df_player_names, df_reports, on='Game ID')

operator = "extract" if args.extract else "map"

# -------- Map --------
if args.provider == 'ollama':
    results_file = f"evaluation/derivation/Q4/results/lotus_Q4_{operator}_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    results_file = f"evaluation/derivation/Q4/results/lotus_Q4_{operator}_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

df_lotus = pd.read_csv(results_file)

df_lotus.rename(columns={"points": "Points", "assists": "Assists", "total_rebounds": "Total rebounds", "steals": "Steals", "blocks": "Blocks"}, inplace=True) 

df = df_labels.merge(df_lotus, on=['Game ID', 'Player Name'], how='left', indicator=True)

df_both = df[df['_merge'] == 'both']
cols = ["Assists", "Total rebounds", "Blocks", "Steals"]

for col in cols:
    xcol, ycol = f"{col}_x", f"{col}_y"
    df_both[f"{col}_match"] = (df_both[xcol].fillna(-1) == df_both[ycol].fillna(-1))

with open('statistics/derivation/Q4.log', 'a') as file:
    for col in cols:
        acc = df_both[f"{col}_match"].mean()
        file.write(f"{col} accuracy: {acc:.2%}" + "\n")

    total_accuracy = df_both[[f"{col}_match" for col in cols]].stack().mean()
    file.write(f"Total accuracy: {total_accuracy:.2%}\n")
    file.write("------------------------------------------------------\n\n\n")



