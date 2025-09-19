import pandas as pd
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs
import time
import wandb

wandb.init(
    project="semantic_operations",
    name="lotus_Q9_filter_default_gemma3_12B_ollama",
    group="semantic selection",
)

lm = LM(model="ollama/gemma3:12b")

lotus.settings.configure(lm=lm)
df_players = pd.read_csv("development/datasets/player_evidence_mine.csv").head(100)
df_players = pd.DataFrame(df_players['Player Name']).rename(columns={'Player Name' : 'player_name'})

user_instruction = "{player_name} is from America."

start = time.time()
df = df_players.sem_filter(user_instruction)
exec_time = time.time() - start

# print(df)
# print(exec_time)

wandb.log({
    "result_table": wandb.Table(dataframe=df),
    "execution_time": exec_time
})
wandb.finish()