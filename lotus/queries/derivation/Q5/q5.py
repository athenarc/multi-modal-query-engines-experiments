import pandas as pd
import lotus
from lotus.models import LM
import time
import wandb
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q5_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

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

df_players =pd.read_csv('datasets/rotowire/player_evidence_mine.csv').dropna(subset=['birth_date', 'nationality']).head(args.size)
df = df_players['Player Name'].to_frame(name='Player Name')


start = time.time()
user_instruction = "What is the birthdate of player {Player Name}? Please return **only** a string containing the birthdate in format DD.MM.YYYY (no reasoning/think)."
df_birthdate = df.sem_map(user_instruction)
df['birth_date'] = df_birthdate['_map']
exec_time = time.time() - start


start = time.time()
user_instruction = "What is the nationality of player {Player Name}? Please return a string containing **only** the nationality (no reaosning/think)."
df_nationality = df.sem_map(user_instruction)
df['nationality'] = df_nationality['_map']
exec_time += time.time() - start


if args.provider == 'ollama':
    output_file = f"evaluation/derivation/Q5/results/lotus_Q5_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider =='vllm':
    output_file = f"evaluation/derivation/Q5/results/lotus_Q5_map_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
    
df.to_csv(output_file)

with open('statistics/derivation/Q5.log', 'a') as file:
    file.write(f"System: Lotus (map)\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Total LLM Calls: {args.size}\n")
    file.write(f"Execution Time: {exec_time:.2f}\n")

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time,
        "total_llm_calls": args.size,
    })
    wandb.finish()