import pandas as pd
from rapidfuzz import process, fuzz
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

player_evi = pd.read_csv("datasets/rotowire/player_evidence_mine.csv")[['Player Name', 'birth_date', 'nationality']].dropna(subset=['birth_date', 'nationality']).head(args.size)

if args.provider == 'ollama' or args.provider == 'transformers':
    results_file = f"evaluation/derivation/Q5/results/blendsql_Q5_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    results_file = f"evaluation/derivation/Q5/results/blendsql_Q5_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

df_blendsql = pd.read_csv(results_file, index_col=0)

df = player_evi.merge(df_blendsql, left_on='Player Name', right_on='Player Name', how='outer')

df["match_birthdate"] = df.apply(
    lambda row: (
        isinstance(row["birth_date_x"], str)
        and isinstance(row["birth_date_y"], str)
        and len(row["birth_date_y"]) <= 30
        and (
            (row["birth_date_y"] in row["birth_date_x"])
            or (row["birth_date_x"] in row["birth_date_y"])
        )
    ),
    axis=1
)

acc_birthdate = df["match_birthdate"].mean()

df["match_nationality"] = df.apply(
    lambda row: (
        isinstance(row["nationality_x"], str)
        and isinstance(row["nationality_y"], str)
        and len(row["nationality_y"]) <= 30
        and (
            row["nationality_y"].lower() in row["nationality_x"].lower()
            or row["nationality_x"].lower() in row["nationality_y"].lower()
            or fuzz.ratio(row["nationality_x"], row["nationality_y"]) >= 70
        )
    ),
    axis=1
)
acc_nationality = df["match_nationality"].mean()

with open('statistics/derivation/Q5.log', 'a') as file:
    file.write(f"Birthdate Accuracy: {acc_birthdate:.2%}" + "\n")
    file.write(f"Nationality Accuracy: {acc_nationality:.2%}" + "\n")

    total_accuracy = df[['match_birthdate', 'match_nationality']].stack().mean()
    file.write(f"Total accuracy: {total_accuracy:.2%}" + "\n")
    file.write("------------------------------------------------------\n\n\n")