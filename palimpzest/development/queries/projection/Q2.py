import palimpzest as pz
from palimpzest.constants import Model
import pandas as pd
from dotenv import load_dotenv
import wandb

load_dotenv()

# UDF to create one record for each player in the list of player names
def explode_player_list(record: dict):
    team_name_list = record.get("team_name_list") or []
    team_name_list = filter(None, team_name_list)
    records = []

    for team_name in team_name_list:
        out_record = {k: v for k, v in record.items()}
        out_record['team_name'] = team_name
        records.append(out_record)
    return records

wandb.init(
    project="semantic_operations",
    name="palimpzest_Q2_gemma_3_12b_ollama",
    group="semantic projection",
)

# updated PZ program
reports = pz.TextFileDataset(id="rotowire_reports", path="development/datasets/rotowire/reports_100/")
reports = reports.sem_add_columns([
    {"name": "team_name_list", "type": list[str], "desc": "Names of teams who played the game, excluding those who are mentioned but did not play."},
])


reports = reports.add_columns(
    udf=explode_player_list,
    cols=[{"name": "team_name", "type": str, "desc": "The name of an NBA team who played in a given game."}],
    cardinality=pz.Cardinality.ONE_TO_MANY,
)
reports = reports.sem_add_columns(
    cols=[
        {"name": "Wins", "type": int, "desc": "The number of Wins that `team_name` has. If the team's Wins are not mentioned in the report, fill the value with -1."},
        {"name": "Losses", "type": int, "desc": "The number of Losses that `team_name` has. If the team's Losses are not mentioned in the report, fill the value with -1."},
        {"name": "Total Points", "type": int, "desc": "The number of Total Points that `team_name` scored. If the team's Total Points are not mentioned in the report, fill the value with -1."},
    ],
    depends_on=["contents", "team_name"],
)
config = pz.QueryProcessorConfig(
    # available_models=[Model.VLLM_LLAMA_3_1_8B_INSTRUCT],
    # api_base="http://localhost:5001/v1",
    available_models=[Model.OLLAMA_GEMMA_3_12B],

)

output = reports.run(config=config)
output_df = output.to_df()
output_df.drop(columns=["team_name_list"], inplace=True)

print(output_df)

# print("Cost: ", output.execution_stats.total_execution_cost)
# output_df.to_csv("mine/rotowire_extraction/player_stats.csv", index=False)

wandb.log({
    "result_table": wandb.Table(dataframe=output_df),
    "execution_time": output.execution_stats.total_execution_time,
    "total_tokens": output.execution_stats.total_tokens
})

wandb.finish()