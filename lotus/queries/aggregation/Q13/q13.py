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
        name="lotus_Q13_aggregation_gemma3_12b_ollama_10000",
        group="semantic aggregation",
    )

lm = LM(model="ollama/gemma3:12b")
lotus.settings.configure(lm=lm)

df_reviews = pd.read_csv("datasets/imdb_reviews/imdb_reviews.csv").head(10000)[['review']]

start = time.time()
df = df_reviews.sem_agg("Do positive or negative reviews prevail? from all {review}. Return 1 for positive or 0 for negative **and only that**.")
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