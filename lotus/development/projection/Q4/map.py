import pandas as pd
import lotus
from lotus.models import LM
import time
import wandb

wandb.init(
    project="semantic_operations",
    name="lotus_map_Q4_llama3.3_70b_ollama",
    group="semantic projection",
)

pd.set_option('display.max_colwidth', None)

lm = LM(model="ollama/llama3.3:70b")
lotus.settings.configure(lm=lm)

df_reports = pd.read_csv("development/datasets/player_evidence_mine.csv")
df = df_reports['Player Name'].to_frame(name='Player Name')

start = time.time()

user_instruction = "What is the birthdate of player {Player Name}? Please return only the birthdate in format DD/MM/YYYY."
df_birthdate = df.sem_map(user_instruction)
df['birthdate'] = df_birthdate['_map']

exec_time = time.time() - start
print(exec_time)

wandb.log({
    "result_table": wandb.Table(dataframe=df),
    "execution_time": exec_time
})

wandb.finish()