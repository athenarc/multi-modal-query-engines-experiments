import pandas as pd
import lotus
from lotus.models import LM
import time
import wandb

wandb.init(
    project="semantic_operations",
    name="lotus_map_Q3_gemma3.12_ollama",
    group="semantic projection",
)

pd.set_option('display.max_colwidth', None)

lm = LM(model="ollama/gemma3:12b")
lotus.settings.configure(lm=lm)

df_players = pd.read_csv("/data/hdd1/users/jzerv/reproduced-systems/Lotus/development/datasets/player_evidence_mine.csv")
df_players.dropna(subset=['nationality'], inplace=True)
df = df_players['Player Name'].to_frame(name='Player Name')

start = time.time()

user_instruction = "What is the nationality of player {Player Name}? Please return only the nationality."
df_nationality = df.sem_map(user_instruction)
df['nationality'] = df_nationality['_map']

exec_time = time.time() - start
print(exec_time)

wandb.log({
    "result_table": wandb.Table(dataframe=df),
    "execution_time": exec_time
})

wandb.finish()