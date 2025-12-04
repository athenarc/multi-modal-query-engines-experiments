import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

df_reports = pd.read_csv('datasets/rotowire/reports_table.csv').rename(columns={'Game_ID': 'Game ID'})
df_player_names = pd.read_csv('datasets/rotowire/player_labels.csv')[['Player Name', 'Game ID', "Points"]].head(args.size)
df = pd.merge(df_player_names, df_reports, on='Game ID')

if args.provider == 'ollama':
    results_file = f"evaluation/derivation/Q1/results/palimpzest_Q1_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    results_file = f"evaluation/derivation/Q1/results/palimpzest_Q1_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

df_pz = pd.read_csv(results_file)

df = df_pz.merge(df_pz, left_on='Player Name', right_on='Player Name')
df["match"] = df.apply(
    lambda row: (
        row["points_x"] == row["points_y"]
    ),
    axis=1
)

accuracy = df['match'].mean()
with open('statistics/derivation/Q1.txt', 'a') as file:
    file.write(f"Accuracy: {df['match'].mean():.2%}" + "\n")
    file.write("------------------------------------------------------\n\n\n")