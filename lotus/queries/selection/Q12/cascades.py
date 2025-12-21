import pandas as pd
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs
import time
import wandb
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=14, const=14, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

helper_model = "meta-llama/Llama-3.1-8B-Instruct"

if args.wandb:
    run_name = f"lotus_Q12_filter_default_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name=run_name,
        group="Selection",
    )

if args.provider == 'ollama':
    lm = LM(args.provider + '/' + args.model)
elif args.provider == 'vllm':
    lm = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)

helper_lm = LM(model=f"hosted_vllm/{helper_model}", api_base="http://localhost:5001/v1", api_key="dummy")

lotus.settings.configure(lm=lm, helper_lm=helper_lm)

df_players = pd.read_csv(f"datasets/rotowire/reports_with_player_names/reports_with_players.csv").head(args.size)

user_instruction = "{Player Name} had 17 points in game {Report}."

cascade_args = CascadeArgs(recall_target=0.9, precision_target=0.9, sampling_percentage=0.2, failure_probability=0.1)

start = time.time()
df = df_players.sem_filter(user_instruction, cascade_args=cascade_args)
exec_time = time.time() - start

output_file = f"evaluation/selection/Q12/results/lotus_Q12_filter_cascades_{args.model.replace(':', '_').replace('/', '_')}_{args.provider}_{args.size}.csv"
df.to_csv(output_file)

with open('statistics/selection/Q12.log', 'a') as file:
    file.write(f"System: Lotus (sem_filter -- optimized)\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Execution Time: {exec_time:.2f}\n")
    # file.write("Total LLM calls: " + str(args.size) + "\n")

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time
    })

    wandb.finish()