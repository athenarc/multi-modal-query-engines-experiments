from datetime import datetime
import pandas as pd
import lotus
from lotus.models import LM
import time
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q1_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

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

df_reports = pd.read_csv('datasets/rotowire/reports_table.csv').rename(columns={'Game_ID': 'Game ID'})
df_player_names = pd.read_csv('datasets/rotowire/player_labels.csv')[['Player Name', 'Game ID']].head(args.size)
df = pd.merge(df_player_names, df_reports, on='Game ID')

start = time.time()

user_instruction = "Return the number of points scored by the {Player Name} in the game described by {Report}. Return **only** an integer."
df_points = df.sem_map(user_instruction)
df['points'] = df_points['_map']

exec_time = time.time() - start

if args.provider == 'ollama':
    output_file = f"evaluation/derivation/Q1/results/lotus_Q1_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider =='vllm':
    output_file = f"evaluation/derivation/Q1/results/lotus_Q1_map_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
    
df.to_csv(output_file)

total_LLM_calls = args.size

with open('statistics/derivation/Q1.log', 'a') as file:
    file.write(f"System: Lotus (sem_map)\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Execution Time: {exec_time:.2f}\n")
    file.write("Total LLM calls: " + str(total_LLM_calls) + "\n")

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time,
        "total_LLM_calls": total_LLM_calls
    })

    wandb.finish()