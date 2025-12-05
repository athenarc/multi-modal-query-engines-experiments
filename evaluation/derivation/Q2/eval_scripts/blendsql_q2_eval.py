import pandas as pd
from rapidfuzz import process, fuzz
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

player_evi = pd.read_csv("datasets/rotowire/player_evidence_mine.csv")[['Player Name', 'birth_place']].dropna(subset=['birth_place']).head(args.size)

if args.provider == 'ollama' or args.provider == 'transformers':
    results_file = f"evaluation/derivation/Q2/results/blendsql_Q2_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    results_file = f"evaluation/derivation/Q2/results/blendsql_Q2_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

blendsql_evi = pd.read_csv(results_file)
blendsql_evi.rename(columns={'player_name': 'Player Name', '_col_1': 'birth_place'}, inplace=True)

df = player_evi.merge(blendsql_evi, left_on='Player Name', right_on='Player Name', how='outer')

df["match"] = df.apply(
    lambda row: (
        isinstance(row["birth_place_x"], str)
        and isinstance(row["birth_place_y"], str)
        and len(row["birth_place_y"]) <= 30
        and (
            row["birth_place_y"].lower() in row["birth_place_x"].lower()
            or row["birth_place_x"].lower() in row["birth_place_y"].lower()
            or fuzz.ratio(row["birth_place_x"], row["birth_place_y"]) >= 70
        )
    ),
    axis=1
)

accuracy = df['match'].mean()
with open('statistics/derivation/Q2.log', 'a') as file:
    file.write(f"Accuracy: {df['match'].mean():.2%}" + "\n")
    file.write("------------------------------------------------------\n\n\n")