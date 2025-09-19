import pandas as pd
import lotus
from lotus.models import LM
import os
import wandb
import time

wandb.init(
    project="semantic_operations",
    name="lotus_NER_TE_extract_llama_3.1_8B_instruct_vllm",
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
    "masked": "A comma-separated list with player names that played in the game. Do not count players that are mentioned but did not play.",
}

new_df = df_reports.sem_extract(input_cols, output_cols) 

df_players = new_df[['Game ID', 'masked']].copy()

df_players['player_name'] = df_players['masked'].str.split(", ")

df_exploded = df_players.explode('player_name', ignore_index=True)

df_players = df_exploded[['Game ID', 'player_name']].copy()

df_merged = pd.merge(df_players, new_df[['Game ID', 'Report']], on='Game ID', how='left')

input_cols = ["Report", "player_name"]
output_cols = {
    "masked_col2": "The number of Points that the {player_name} scored or -1 if not mentioned.",
    "masked_col3": "The number of Assists that the {player_name} scored or -1 if not mentioned.",
    "masked_col4": "The total number of rebounds that the {player_name} had or -1 if not mentioned",
    "masked_col5": "The steals that the {player_name} had or -1 if not mentioned",
    "masked_col6": "The blocks that the {player_name} had or -1 if not mentioned",
    # "masked_col7": "The defensive rebounds that the {player_name} had or 0 if not mentioned",
    # "masked_col8": "The offensive rebounds that the {player_name} had or 0 if not mentioned",
    # "masked_col9": "The personal fouls that the {player_name} had or 0 if not mentioned.",
    # "masked_col10": "The turnovers that the {player_name} had or 0 if not mentioned.",
    # "masked_col11": "The field goals made by {player_name} or 0 if not mentioned",
    # "masked_col12": "The field goals attempted by {player_name} or 0 if not mentioned",
    # "masked_col13": "The field goal percentage of {player_name} or 0 if not mentioned",
    # "masked_col14": "The free throws made by {player_name} or 0 if not mentioned",
    # "masked_col15": "The free throws attempted by {player_name} or 0 if not mentioned",
    # "masked_col16": "The free throw percentage of {player_name} or 0 if not mentioned",
    # "masked_col17": "The three pointers attempted by {player_name} or 0 if not mentioned",
    # "masked_col18": "The three pointers made by {player_name} or 0 if not mentioned",
    # "masked_col19": "The minutes played that the {player_name} had or 0 if not mentioned."
}
new_df = df_merged.sem_extract(input_cols, output_cols, extract_quotes=False)

# new_df.rename({"masked_col2": "points", "masked_col3": "assists", "masked_col4": "rebounds", "masked_col5": "steals", "masked_col6": "blocks", "masked_col7": "defensive_rebounds", "masked_col8": "offensive_rebounds", "masked_col9": "personal_fouls", "masked_col10": "turnovers", "masked_col11": "field_goals_made", "masked_col12": "field_goals_attempted", "masked_col13": "field_goal_percentage", "masked_col14": "free_throws_made", "masked_col15": "free_throws_attempted", "masked_col16": "free_throw_percentage", "masked_col17": "three_pointers_attempted", "masked_col18": "three_pointers_made", "masked_col19": "minutes_played"}, inplace=True)
# new_df = new_df.rename(columns={"masked_col2": "points", "masked_col3": "assists", "masked_col4": "rebounds", "masked_col5": "steals", "masked_col6": "blocks", "masked_col7": "defensive_rebounds", "masked_col8": "offensive_rebounds", "masked_col9": "personal_fouls", "masked_col10": "turnovers", "masked_col11": "field_goals_made", "masked_col12": "field_goals_attempted", "masked_col13": "field_goal_percentage", "masked_col14": "free_throws_made", "masked_col15": "free_throws_attempted", "masked_col16": "free_throw_percentage", "masked_col17": "three_pointers_attempted", "masked_col18": "three_pointers_made", "masked_col19": "minutes_played"})
# df = new_df[['Game ID', 'points', 'assists', 'rebounds', 'steals', 'blocks', 'defensive_rebounds', 'offensive_rebounds', 'personal_fouls', 'turnovers', 'field_goals_made', 'field_goals_attempted', 'field_goal_percentage', 'free_throws_made', 'free_throws_attempted', 'free_throw_percentage', 'three_pointers_attempted', 'three_pointers_made', 'minutes_played']]
new_df.rename({"masked_col2": "points", "masked_col3": "assists", "masked_col4": "rebounds", "masked_col5": "steals", "masked_col6": "blocks"}, inplace=True)
new_df = new_df.rename(columns={"masked_col2": "points", "masked_col3": "assists", "masked_col4": "rebounds", "masked_col5": "steals", "masked_col6": "blocks"})
df = new_df[['Game ID', 'player_name', 'points', 'assists', 'rebounds', 'steals', 'blocks']]
exec_time = time.time() - start

wandb.log({
    "result_table": wandb.Table(dataframe=df),
    "execution_time": exec_time
})

wandb.finish()