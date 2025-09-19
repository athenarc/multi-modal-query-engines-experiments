import pandas as pd
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs
import time
import wandb

wandb.init(
    project="semantic_operations",
    name="lotus_Q10_filter_default_gemma3_12B_ollama_llama8b_vllm",
    group="semantic selection",
)

lm = LM(model="ollama/gemma3:12b")

lotus.settings.configure(lm=lm)
df_teams = pd.read_csv("development/datasets/team_evidence.csv")
df_teams = pd.DataFrame(df_teams['Team Name']).rename(columns={'Team Name' : 'team_name'})

user_instruction = "{team_name} founded before 1970."

start = time.time()
df = df_teams.sem_filter(user_instruction)
exec_time = time.time() - start

# print(df)
# print(exec_time)

wandb.log({
    "result_table": wandb.Table(dataframe=df),
    "execution_time": exec_time
})
wandb.finish()