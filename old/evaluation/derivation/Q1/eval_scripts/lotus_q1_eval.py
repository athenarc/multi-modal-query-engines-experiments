import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
parser.add_argument("-e", "--extract", action='store_true', help="Evaluate extract instead of map")
args = parser.parse_args()

df_reports = pd.read_csv('datasets/rotowire/reports_table.csv').rename(columns={'Game_ID': 'Game ID'})
df_player_names = pd.read_csv('datasets/rotowire/player_labels.csv')[['Player Name', 'Game ID', "Points"]].head(args.size)
df = pd.merge(df_player_names, df_reports, on='Game ID')

operator = "extract" if args.extract else "map"

if args.provider == 'ollama':
    results_file = f"evaluation/derivation/Q1/results/lotus_Q1_{operator}_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    results_file = f"evaluation/derivation/Q1/results/lotus_Q1_{operator}_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

    df_lotus = pd.read_csv(results_file)

    df = df_lotus.merge(df_lotus, left_on='Player Name', right_on='Player Name')
    df["match"] = df.apply(
        lambda row: (
            row["points_x"] == row["points_y"]
        ),
        axis=1
    )

    accuracy = df['match'].mean()
    with open('statistics/derivation/Q1.log', 'a') as file:
        file.write(f"Accuracy: {df['match'].mean():.2%}" + "\n")
        file.write("------------------------------------------------------\n\n\n")
