import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
parser.add_argument("--system", nargs='?', default='lotus', const='lotus', type=str, help="The system that is being evaluated")
args = parser.parse_args()

system_results = "lotus_Q6_map" if args.system == 'lotus' else f"{args.system}_Q6"

team_labels = pd.read_csv("datasets/rotowire/team_labels.csv")[['Game ID', 'Total points', 'Team Name', 'Wins', 'Losses']].head(args.size * 2).fillna(-1)

if args.provider == 'ollama' or args.provider == 'transformers':
    results_file = f"evaluation/derivation/Q6/results/{system_results}_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    results_file = f"evaluation/derivation/Q6/results/{system_results}_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"


winners = team_labels.loc[team_labels.groupby("Game ID")["Total points"].idxmin(), ["Game ID", "Wins", "Losses"]]

df_results = pd.read_csv(results_file, index_col=0)[['Game ID', 'Wins', "Losses"]].fillna(-1)
df = df_results.merge(winners, on='Game ID', how='outer')

df["match_wins"] = df.apply(
    lambda row: (
        row["Wins_x"] == row["Wins_y"]
    ),
    axis=1
)
acc_wins = df['match_wins'].mean()

df["match_losses"] = df.apply(
    lambda row: (
        row["Losses_x"] == row["Losses_y"]
    ),
    axis=1
)
acc_losses = df['match_losses'].mean()

with open('statistics/derivation/Q6.log', 'a') as file:
    file.write(f"Wins accuracy: {acc_wins:.2%}" + "\n")
    file.write(f"Losses accuracy: {acc_losses:.2%}" + "\n")

    total_accuracy = df[['match_wins', 'match_losses']].stack().mean()
    file.write(f"Total accuracy: {total_accuracy:.2%}" + "\n")
    file.write("------------------------------------------------------\n\n\n")