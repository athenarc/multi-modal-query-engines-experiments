import pandas as pd
import lotus
from lotus.models import LM
import wandb
import time

wandb.init(
    project="semantic_operations",
    name="lotus_Q11_join_default_gemma3_12b_ollama",
    group="semantic join",
)

df_players = pd.read_csv("development/datasets/player_evidence_mine.csv").head(50)[['Player Name']]
df_teams = pd.read_csv("development/datasets/team_evidence.csv")[['Team Name']]

lm = LM(model="ollama/llama3.3:70b")
lotus.settings.configure(lm=lm)

instruction = "The player {Player Name:left} was playing for team {Team Name:right} in 2015."
start = time.time()
df = df_players.sem_join(df_teams, instruction)
exec_time = time.time() - start

wandb.log({
    "result_table": wandb.Table(dataframe=df),
    "execution_time": exec_time
})

wandb.finish()