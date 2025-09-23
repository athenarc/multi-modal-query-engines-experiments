import pandas as pd
import time
import wandb
import argparse

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMJoin

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
args = parser.parse_args()

if args.wandb:
    wandb.init(
        project="semantic_operations",
        name="blendsql_q11_gemma3_12b_ollama",
        group="semantic join",
    )

players_df = pd.read_csv('datasets/rotowire/player_evidence_mine.csv').head(50)[['Player Name']].rename(columns={'Player Name' : 'player_name'})

teams_df = pd.read_csv('datasets/rotowire/team_evidence.csv')[['Team Name']].rename(columns={'Team Name' : 'team_name'})

db = {
    "Players": pd.DataFrame(players_df),
    "Teams": pd.DataFrame(teams_df)
}

bsql = BlendSQL(
    db=db,
    model=LiteLLM("ollama/gemma3:12b", config={"timeout": 50000}),
    ingredients={LLMJoin}
)

start = time.time()
smoothie = bsql.execute(
    """
        SELECT *
        FROM Players p
        JOIN Teams t ON {{
            LLMJoin(
                'p.player_name',
                't.team_name',
                question='The player was playing for the team in 2015.',
            )
        }} 
    """,
    infer_gen_constraints=True,
)

exec_time = time.time()-start

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=smoothie.df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", smoothie.df)
    print("Execution time: ", exec_time)

