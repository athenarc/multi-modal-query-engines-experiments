import pandas as pd
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs
import time
import wandb

wandb.init(
    project="semantic_operations",
    name="lotus_Q6_filter_default_gemma3_12B_ollama",
    group="semantic selection",
)

lm = LM(model="ollama/gemma3:12b")

lotus.settings.configure(lm=lm)
df_players = pd.read_csv("/data/hdd1/users/jzerv/reproduced-systems/Lotus/reports_with_players_100.csv")
df_players = df_players[df_players["Game ID"] < 14] # Keep total of ~100 entries

user_instruction = "{Player Name} had 17 points in game {Report}."

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