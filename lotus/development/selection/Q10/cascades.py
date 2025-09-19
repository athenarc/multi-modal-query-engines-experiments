import pandas as pd
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs
import time
import wandb

wandb.init(
    project="semantic_operations",
    name="lotus_Q10_filter_cascades_gemma3_12B_ollama_llama8b_vllm",
    group="semantic selection",
)

lm = LM(model="ollama/gemma3:12b")
helper_lm = LM(model="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct", api_base="http://localhost:5001/v1", api_key="dummy")

lotus.settings.configure(lm=lm, helper_lm=helper_lm)
df_teams = pd.read_csv("/data/hdd1/users/jzerv/reproduced-systems/Lotus/development/datasets/team_evidence.csv")
df_teams = pd.DataFrame(df_teams['Team Name']).rename(columns={'Team Name' : 'team_name'})

user_instruction = "{team_name} founded before 1970."

cascade_args = CascadeArgs(recall_target=0.9, precision_target=0.9, sampling_percentage=0.2, failure_probability=0.1)

start = time.time()
df = df_teams.sem_filter(user_instruction, cascade_args=cascade_args)
exec_time = time.time() - start

# print(df)
# print(exec_time)

wandb.log({
    "result_table": wandb.Table(dataframe=df),
    "execution_time": exec_time
})
wandb.finish()