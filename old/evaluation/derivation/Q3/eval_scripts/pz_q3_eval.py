import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

team_labels = pd.read_csv("datasets/rotowire/team_labels.csv")[['Game ID', 'Team Name', 'Total points']].head(args.size * 2).fillna(-1)

if args.provider == 'ollama' or args.provider == 'transformers':
    results_file = f"evaluation/derivation/Q3/results/palimpzest_Q3_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    results_file = f"evaluation/derivation/Q3/results/palimpzest_Q3_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"


winners = team_labels.loc[team_labels.groupby("Game ID")["Total points"].idxmax(), ["Game ID", "Total points"]].rename(columns={"Total points": "points"})

df_pz = pd.read_csv(results_file, index_col=0)[['Game ID', 'points']].fillna(-1)
df = df_pz.merge(winners, on='Game ID', how='outer', suffixes=('_pred', '_gt'))

df["match"] = df.apply(
    lambda row: (
        pd.to_numeric(row["points_pred"], errors='coerce') == pd.to_numeric(row["points_gt"], errors='coerce')
    ) if (pd.to_numeric(row["points_pred"], errors='coerce') is not np.nan and pd.to_numeric(row["points_gt"], errors='coerce') is not np.nan) else False,
    axis=1
)

accuracy = df['match'].mean()
with open('statistics/derivation/Q3.log', 'a') as file:
    file.write(f"Accuracy: {df['match'].mean():.2%}" + "\n")
    file.write("------------------------------------------------------\n\n\n")