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
    run_name = f"lotus_Q4_extract_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

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

df_reports = pd.read_csv("datasets/rotowire/reports_table.csv").head(args.size).rename(columns={'Game_ID' : 'Game ID'})
df_players = pd.read_csv('datasets/rotowire/player_labels.csv')[['Player Name', 'Game ID']].head(args.size)
df = pd.merge(df_players, df_reports, on='Game ID')

input_cols = ["Report", "Player Name"]

start = time.time()
# A description can be specified for each output column
output_cols = {
    "masked_col1": "The number of Assists that the {Player Name} scored or -1 if not mentioned.",
    "masked_col2": "The total number of rebounds that the {Player Name} had or -1 if not mentioned",
    "masked_col3": "The steals that the {Player Name} had or -1 if not mentioned",
    "masked_col4": "The blocks that the {Player Name} had or -1 if not mentioned"
}

new_df = df.sem_extract(input_cols, output_cols, extract_quotes=False)

new_df.rename(columns={"masked_col1": "Assists", "masked_col2": "Total rebounds", "masked_col3": "Steals", "masked_col4": "Blocks"}, inplace=True)
df = new_df[['Game ID', 'Player Name', 'Assists', 'Total rebounds', 'Blocks', 'Steals']]


exec_time = time.time() - start

if args.provider == 'ollama':
    output_file = f"evaluation/derivation/Q4/results/lotus_Q4_extract_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider =='vllm':
    output_file = f"evaluation/derivation/Q4/results/lotus_Q4_extract_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
    
df.to_csv(output_file)

with open('statistics/derivation/Q4.log', 'a') as file:
    file.write(f"System: Lotus (sem_extrat)\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Total LLM calls: " + str(args.size) + "\n")
    file.write(f"Execution Time: {exec_time:.2f}\n")

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time
    })

    wandb.finish()