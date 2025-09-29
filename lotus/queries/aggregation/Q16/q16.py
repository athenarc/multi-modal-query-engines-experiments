import pandas as pd
import lotus
from lotus.models import LM
import time
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q16_aggregation_{args.model.replace(':', '_')}_{args.provider}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic aggregation",
    )

if (args.provider == 'ollama'):
    model = args.provider + '/' + args.model

lm = LM(model=model)
lotus.settings.configure(lm=lm)

df_reviews = pd.read_csv("datasets/rotowire/reports_table.csv")

start = time.time()
df = df_reviews.sem_agg("Summarize the performance of Giannis Antetokoumpo in all of his matches described by {Report}.")
exec_time = time.time() - start

if args.wandb:
    wandb.log({
        "result": wandb.Table(dataframe=df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", df)
    print("Execution time: ", exec_time)