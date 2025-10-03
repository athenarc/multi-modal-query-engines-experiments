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
    run_name = f"lotus_Q4_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic projection",
    )

if args.provider == 'ollama':
    lm = LM(args.provider + '/' + args.model)
elif args.provider == 'vllm':
    lm = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)

lotus.settings.configure(lm=lm)

df_reports = pd.read_csv("datasets/rotowire/player_evidence_mine.csv").head(args.size)
df = df_reports['Player Name'].to_frame(name='Player Name')

start = time.time()

user_instruction = "What is the birthdate of player {Player Name}? Please return only the birthdate in format DD/MM/YYYY."
df_birthdate = df.sem_map(user_instruction)
df['birthdate'] = df_birthdate['_map']

exec_time = time.time() - start

if args.wandb:
    if args.provider == 'ollama':
        output_file = f"evaluation/projection/Q4/results/lotus_Q4_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
    elif args.provider =='vllm':
        output_file = f"evaluation/projection/Q4/results/lotus_Q4_map_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
        
    df.to_csv(output_file)

    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", df)
    print("Execution time: ", exec_time)