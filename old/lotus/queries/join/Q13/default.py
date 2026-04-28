from datetime import datetime
import pandas as pd
import lotus
from lotus.models import LM
import wandb
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=50, const=50, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q13_join_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name=run_name,
        group="Join",
    )

df_text = pd.read_csv("datasets/banking_data/banking_data_test.csv")[['text']].head(args.size)
df_categories = pd.read_csv("datasets/banking_data/categories.csv").head(63)

if args.provider == 'ollama':
    model = LM(args.provider + '/' + args.model)
elif args.provider == 'vllm':
    model = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)

lotus.settings.configure(lm=model)

instruction = "The online banking query {text:left} maps the category-intent {category:right}."
start = time.time()
df = df_text.sem_join(df_categories, instruction)
exec_time = time.time() - start

output_file = f"evaluation/join/Q13/results/lotus_Q13_join_{args.model.replace(':', '_').replace('/', '_')}_{args.provider}_{args.size}.csv"
df.to_csv(output_file)
with open('statistics/join/Q13.log', 'a') as file:
    file.write(f"System: Lotus (sem_join -- default)\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Execution Time: {exec_time:.2f}\n")
    file.write("Total LLM calls: " + str(args.size * 63) + "\n")


if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time,
        "total_LLM_calls": args.size*63
    })

    wandb.finish()