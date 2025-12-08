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
    run_name = f"lotus_Q7_extract_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name= run_name,
        group="Derivation",
    )

if args.provider == 'ollama':
    lm = LM(args.provider + '/' + args.model)
elif args.provider == 'vllm':
    lm = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)

lotus.settings.configure(lm=lm)
df_reports = pd.read_csv("datasets/rotowire/reports_table.csv").head(args.size).rename(columns={'Game_ID' : 'Game ID'})

input_cols = ["Report"]

start = time.time()
# A description can be specified for each output column
output_cols = {
    "masked": "A comma-separated list with player names that played in the game. Do not count players that are mentioned but did not play.",
}

new_df = df_reports.sem_extract(input_cols, output_cols) 

df_players = new_df[['Game ID', 'masked']].copy()

df_players['Player Name'] = df_players['masked'].str.split(", ")

df_exploded = df_players.explode('Player Name', ignore_index=True)

df_players = df_exploded[['Game ID', 'Player Name']].copy()

df_merged = pd.merge(df_players, new_df[['Game ID', 'Report']], on='Game ID', how='left')

input_cols = ["Report", "Player Name"]
output_cols = {
    "masked_col2": "The number of Points that the {Player Name} scored or -1 if not mentioned.",
    "masked_col3": "The number of Assists that the {Player Name} scored or -1 if not mentioned.",
    "masked_col4": "The total number of rebounds that the {Player Name} had or -1 if not mentioned",
    "masked_col5": "The steals that the {Player Name} had or -1 if not mentioned",
    "masked_col6": "The blocks that the {Player Name} had or -1 if not mentioned",
    # "masked_col7": "The defensive rebounds that the {Player Name} had or 0 if not mentioned",
    # "masked_col8": "The offensive rebounds that the {Player Name} had or 0 if not mentioned",
    # "masked_col9": "The personal fouls that the {Player Name} had or 0 if not mentioned.",
    # "masked_col10": "The turnovers that the {Player Name} had or 0 if not mentioned.",
    # "masked_col11": "The field goals made by {Player Name} or 0 if not mentioned",
    # "masked_col12": "The field goals attempted by {Player Name} or 0 if not mentioned",
    # "masked_col13": "The field goal percentage of {Player Name} or 0 if not mentioned",
    # "masked_col14": "The free throws made by {Player Name} or 0 if not mentioned",
    # "masked_col15": "The free throws attempted by {Player Name} or 0 if not mentioned",
    # "masked_col16": "The free throw percentage of {Player Name} or 0 if not mentioned",
    # "masked_col17": "The three pointers attempted by {Player Name} or 0 if not mentioned",
    # "masked_col18": "The three pointers made by {Player Name} or 0 if not mentioned",
    # "masked_col19": "The minutes played that the {Player Name} had or 0 if not mentioned."
}
new_df = df_merged.sem_extract(input_cols, output_cols, extract_quotes=False)

new_df = new_df.rename(columns={"masked_col2": "points", "masked_col3": "assists", "masked_col4": "Total rebounds", "masked_col5": "steals", "masked_col6": "blocks"})
df = new_df[['Game ID', 'Player Name', 'points', 'assists', 'Total rebounds', 'steals', 'blocks']]
exec_time = time.time() - start

if args.provider == 'ollama':
    output_file = f"evaluation/derivation/Q7/results/lotus_Q7_extract_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider =='vllm':
    output_file = f"evaluation/derivation/Q7/results/lotus_Q7_extract_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
df.to_csv(output_file)
    
num_extraction_attributes = 5  # points, assists, total_rebounds, blocks, steals

LLM_calls_for_rows = df_reports.shape[0]
LLM_calls_for_columns = df_reports.shape[0] * num_extraction_attributes
total_LLM_calls = LLM_calls_for_rows + LLM_calls_for_columns

with open('statistics/derivation/Q7.log', 'a') as file:
    file.write(f"System: Lotus (sem_extract)\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write("Execution Time: " + str(exec_time) + "\n\n")
    file.write("LLM calls for rows: " + str(LLM_calls_for_rows) + "\n")
    file.write("LLM calls for columns: " + str(LLM_calls_for_columns) + "\n")
    file.write("Total LLM calls: " + str(total_LLM_calls) + "\n")


if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time,
        "LLM_calls_for_rows": LLM_calls_for_rows,
        "LLM_calls_for_columns": LLM_calls_for_columns,
        "total_LLM_calls": total_LLM_calls
    })
    wandb.finish()
