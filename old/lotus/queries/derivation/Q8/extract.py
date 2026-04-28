import pandas as pd
import lotus
from lotus.models import LM
import os
import wandb
import time
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q8_extract_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name=run_name,
        group="Derivation",
    )

if args.provider == 'ollama':
    lm = LM(args.provider + '/' + args.model)
elif args.provider == 'vllm':
    lm = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)

lotus.settings.configure(lm=lm)

df_reports = pd.read_csv("datasets/rotowire/reports_table.csv").rename(columns={'Game_ID' : 'Game ID'})
missing_game_ids = [8, 39, 68, 82, 122, 123, 150, 155, 192, 199, 211, 214, 255, 267, 274, 290, 294, 313, 330, 343, 345, 363, 379, 391, 398, 423, 439, 472, 499, 500, 534, 558, 562, 565, 568, 570, 644, 645, 668, 681, 721]
df = df_reports[~df_reports['Game ID'].isin(missing_game_ids)]  # Remove Game IDs that are not present in the team labels file
df_reports = df.head(args.size)

input_cols = ["Report"]

start = time.time()
# A description can be specified for each output column
output_cols = {
    "masked": "A comma-separated list with team names that played in the game. Do not count teams that are mentioned but did not play.",
}

new_df = df_reports.sem_extract(input_cols, output_cols) 

df_players = new_df[['Game ID', 'masked']].copy()

df_players['Team Name'] = df_players['masked'].str.split(", ")

df_exploded = df_players.explode('Team Name', ignore_index=True)

df_players = df_exploded[['Game ID', 'Team Name']].copy()

df_merged = pd.merge(df_players, new_df[['Game ID', 'Report']], on='Game ID', how='left')

input_cols = ["Report", "Team Name"]
output_cols = {
    "masked_col2": "The number of Wins that the {Team Name} has or -1 if not mentioned.",
    "masked_col3": "The number of Losses that the {Team Name} has or -1 if not mentioned.",
    "masked_col4": "The number of Total Points that the {Team Name} scored or -1 if not mentioned",
}
new_df = df_merged.sem_extract(input_cols, output_cols, extract_quotes=False)

new_df = new_df.rename(columns={"masked_col2": "wins", "masked_col3": "losses", "masked_col4": "total_points"})
df = new_df[['Game ID', 'Team Name', 'wins', 'losses', 'total_points']]
exec_time = time.time() - start

if args.provider == 'ollama':
    output_file = f"evaluation/derivation/Q8/results/lotus_Q8_extract_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider =='vllm':
    output_file = f"evaluation/derivation/Q8/results/lotus_Q8_extract_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
df.to_csv(output_file)
    
num_extraction_attributes = 3  # Wins, Losses, Total Points

total_LLM_calls = df_reports.shape[0]
# processed_rows = total_LLM_calls

with open('statistics/derivation/Q8.log', 'a') as file:
    file.write(f"System: Lotus (sem_extract)\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write("Execution Time: " + str(exec_time) + "\n")
    file.write("Total LLM calls: " + str(total_LLM_calls) + "\n")
    # file.write("Processed Rows: " + str(processed_rows) + "\n")

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time
        # "Total LLM calls": total_LLM_calls,
        # "Processed Rows": processed_rows
    })
    wandb.finish()