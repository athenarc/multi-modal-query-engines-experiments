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
        name="lotus_Q14_aggregation_gemma3_12b_ollama",
        group="semantic aggregation",
    )

lm = LM(model="ollama/gemma3:12b")
lotus.settings.configure(lm=lm)

df_reviews = pd.read_csv("datasets/enron_emails/enron_emails_shuffled_10000.csv")[['Message']]

start = time.time()
df = df_reviews.sem_agg("Do spam or non-spam emails prevail? from all {Message}. Return 1 for spam or 0 for non-spam **and only that**.")
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