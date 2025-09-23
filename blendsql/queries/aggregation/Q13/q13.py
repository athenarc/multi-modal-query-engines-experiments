import pandas as pd
import time
import wandb
import argparse

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMQA

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
args = parser.parse_args()

if args.wandb:
    wandb.init(
        project="semantic_operations",
        name="blendsql_q13_aggregation_gemma3_12b_ollama",
        group="semantic aggregation",
    )

df_reviews = pd.read_csv("datasets/imdb_reviews/imdb_reviews.csv").head(10000)[['review']]

db = {
    "Reviews": pd.DataFrame(df_reviews)
}

bsql = BlendSQL(
    db=db,
    model=LiteLLM("ollama/gemma3:12b", config={"timeout": 50000}),
    ingredients={LLMQA}
)

start = time.time()
smoothie = bsql.execute(
    """
        SELECT {{
            LLMQA(
                'Do positive or negative reviews prevail? Return 1 for positive or 0 for negative **and only that**.',
                context=Reviews.review
            )
        }} AS Answer
    """,
    infer_gen_constraints=True,
)

exec_time = time.time()-start

if args.wandb:
    wandb.log({
        "result": wandb.Table(dataframe=smoothie.df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", smoothie.df)
    print("Execution time: ", exec_time)
