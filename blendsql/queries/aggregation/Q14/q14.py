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
        name="blendsql_q14_aggregation_gemma3_12b_ollama",
        group="semantic aggregation",
    )

df_emails = pd.read_csv("datasets/enron_emails/enron_emails_shuffled_10000.csv")[['Subject', 'Message']]

db = {
    "Emails": df_emails
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
                'Do spam or non-spam emails prevail? from all emails? Return 1 for spam or 0 for non-spam **and only that**.',
                context=Emails.Message
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