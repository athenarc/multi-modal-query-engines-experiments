import palimpzest as pz
from palimpzest.constants import Model
import pandas as pd
from dotenv import load_dotenv
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=14, const=14, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

model = getattr(Model, f"{args.provider.upper()}_{args.model.replace(':', '_').replace('/', '_').replace('.', '_').replace('-', '_').upper()}")

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

if args.wandb:
    run_name=f"palimpzest_Q6_filter_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic selection",
    )

reports = pz.TextFileDataset(id="rotowire_reports", path=f"datasets/rotowire/reports/{args.size}")
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
    available_models=[model],
)

output = reports.run(config=config)
output_df = output.to_df()
output_df.drop(columns=["player_name_list"], inplace=True)

if args.wandb:
    if args.provider == 'ollama':
        output_file = f"evaluation/selection/Q6/results/palimpzest_Q6_filter_{args.model.replace(':', '_')}_{args.provider}.csv"
    elif args.provider == 'vllm':
        output_file = f"evaluation/selection/Q6/results/palimpzest_Q6_filter_{args.model.replace('/', '_')}_{args.provider}.csv"
    
    output_df.to_csv(output_file)

    wandb.log({
        "result_table": wandb.Table(dataframe=output_df),
        "execution_time": output.execution_stats.total_execution_time,
        "total_tokens": output.execution_stats.total_tokens
    })

    wandb.finish()
else:
    print("Result:\n\n", output_df)
    print("Execution time: ", output.executions_stats.total_execution_time)