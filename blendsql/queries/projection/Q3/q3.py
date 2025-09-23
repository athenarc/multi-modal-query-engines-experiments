import pandas as pd
import time
import wandb
import argparse

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMMap, LLMQA, LLMJoin

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
args = parser.parse_args()

if args.wandb:
    wandb.init(
        project="semantic_operations",
        name="blendsql_Q3_gemma3:12b_ollama",
        group="semantic projection",
)

# Load reports
reports = pd.read_csv('datasets/rotowire/player_evidence_mine.csv').dropna(subset=['nationality'])
reports.rename(columns={"Player Name": "player_name"}, inplace=True)
players = {
    "Players" : pd.DataFrame(reports['player_name'])
}

# Prepare our BlendSQL connection
bsql = BlendSQL(
    db=players,
    # model=TransformersLLM(
    #     "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    #     config={"device_map": "auto"},
    #     caching=False,
    # ),
    model=LiteLLM("ollama/gemma3:12b", config={"timeout": 50000}),
    # model=LiteLLM("qwen3:30b-a3b-instruct-2507-q4_K_M", config={"timeout": 50000}),
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()

smoothie = bsql.execute(
   """
    SELECT Players.player_name, {{
        LLMMAP(
            'Return the nationality of the player.',
            Players.player_name,
        )
    }}
    FROM Players
    """,
    infer_gen_constraints=True,
)

exec_time = time.time() - start
print(smoothie.df)

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=smoothie.df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", smoothie.df)
    print("Execution time: ", exec_time)