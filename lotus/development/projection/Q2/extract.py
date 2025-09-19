import pandas as pd
import lotus
from lotus.models import LM
import os
import wandb
import time

wandb.init(
    project="semantic_operations",
    name="lotus_NER_TE_Q2_extract_llama_3.1_8B_instruct_vllm",
    group="semantic projection",
)

lm = LM(
    model="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct", 
    api_base="http://localhost:5001/v1",
    api_key="dummy")

lotus.settings.configure(lm=lm)
df_reports = pd.read_csv("/data/hdd1/users/jzerv/reproduced-systems/Lotus/reports_table.csv").head(100)

input_cols = ["Report"]

start = time.time()
# A description can be specified for each output column
output_cols = {
    "masked": "A comma-separated list with team names that played in the game. Do not count teams that are mentioned but did not play.",
}

new_df = df_reports.sem_extract(input_cols, output_cols) 

df_players = new_df[['Game ID', 'masked']].copy()

df_players['team_name'] = df_players['masked'].str.split(", ")

df_exploded = df_players.explode('team_name', ignore_index=True)

df_players = df_exploded[['Game ID', 'team_name']].copy()

df_merged = pd.merge(df_players, new_df[['Game ID', 'Report']], on='Game ID', how='left')

input_cols = ["Report", "team_name"]
output_cols = {
    "masked_col2": "The number of Wins that the {team_name} has or -1 if not mentioned.",
    "masked_col3": "The number of Losses that the {team_name} has or -1 if not mentioned.",
    "masked_col4": "The number of Total Points that the {team_name} scored or -1 if not mentioned",
}
new_df = df_merged.sem_extract(input_cols, output_cols, extract_quotes=False)

new_df = new_df.rename(columns={"masked_col2": "wins", "masked_col3": "losses", "masked_col4": "total_points"})
df = new_df[['Game ID', 'team_name', 'wins', 'losses', 'total_points']]
exec_time = time.time() - start

wandb.log({
    "result_table": wandb.Table(dataframe=df),
    "execution_time": exec_time
})

wandb.finish()