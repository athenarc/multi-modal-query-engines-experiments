from datetime import datetime
import pandas as pd
import lotus
from lotus.models import LM
import time
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=1000, const=1000, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q16_aggregation_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name=run_name,
        group="Aggregation",
    )

if args.provider == 'ollama':
    model = LM(args.provider + '/' + args.model)
elif args.provider == 'vllm':
    model = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)

lotus.settings.configure(lm=model)

df_reviews = pd.read_csv("datasets/imdb_reviews/imdb_reviews.csv").head(args.size)[['review']]

start = time.time()
answer = df_reviews.sem_agg("Count all the positive {review}. Return **only** an integer.")
exec_time = time.time() - start

with open('statistics/aggregation/Q16.log', 'a') as file:
    file.write(f"System: Lotus \n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Execution Time: {exec_time:.2f}\n")
    file.write(f"Answer: {answer}\n")
    # file.write("Total LLM calls: " + str(total_LLM_calls) + "\n")


if args.wandb:
    wandb.log({
        "result": wandb.Table(dataframe=answer),
        "execution_time": exec_time
    })
    wandb.finish()