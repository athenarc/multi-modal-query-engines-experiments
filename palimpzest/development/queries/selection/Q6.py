import palimpzest as pz
from palimpzest.constants import Model
import pandas as pd
from dotenv import load_dotenv
import wandb

load_dotenv()

# UDF to create one record for each player in the list of player names
def explode_player_list(record: dict):
    player_name_list = record.get("player_name_list") or []
    player_name_list = filter(None, player_name_list)
    records = []

    for player_name in player_name_list:
        out_record = {k: v for k, v in record.items()}
        out_record['player_name'] = player_name
        records.append(out_record)
    return records

wandb.init(
    project="semantic_operations",
    name="palimpzest_Q6_gemma3_12b_ollama",
    group="semantic selection",
)

reports = pz.TextFileDataset(id="rotowire_reports", path="development/datasets/rotowire/reports/reports_14")
reports = reports.sem_add_columns([
    {"name": "player_name_list", "type": list[str], "desc": "Names of players who played the game, excluding those who are mentioned but did not play."},
])

reports = reports.add_columns(
    udf=explode_player_list,
    cols=[{"name": "player_name", "type": str, "desc": "The name of an NBA player who played in a given game."}],
    cardinality=pz.Cardinality.ONE_TO_MANY,
)

reports = reports.sem_filter("The player specified by the `player_name` field scored 17 points.")

config = pz.QueryProcessorConfig(
    available_models=[Model.OLLAMA_GEMMA_3_12B],
)

output = reports.run(config=config)
output_df = output.to_df()
output_df.drop(columns=["player_name_list"], inplace=True)

print(output_df)

# print("Cost: ", output.execution_stats.totxwal_execution_cost)
# output_df.to_csv("mine/rotowire_extraction/player_stats.csv", index=False)

wandb.log({
    "result_table": wandb.Table(dataframe=output_df),
    "execution_time": output.execution_stats.total_execution_time,
    "total_tokens": output.execution_stats.total_tokens
})

wandb.finish()