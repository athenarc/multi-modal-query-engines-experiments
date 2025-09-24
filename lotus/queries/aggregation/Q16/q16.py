import pandas as pd
import lotus
from lotus.models import LM
import time
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
args = parser.parse_args()

if args.wandb:
    wandb.init(
        project="semantic_operations",
        name="lotus_Q16_aggregation_gemma3_12b_ollama",
        group="semantic aggregation",
    )

lm = LM(model="ollama/gemma3:12b")
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