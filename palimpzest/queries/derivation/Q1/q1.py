import palimpzest as pz
from palimpzest.constants import Model
import pandas as pd
from dotenv import load_dotenv
import wandb
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

model = getattr(Model, f"{args.provider.upper()}_{args.model.replace(':', '_').replace('/', '_').replace('.', '_').replace('-', '_').upper()}")

load_dotenv()
if args.wandb:
    run_name=f"palimpzest_Q1_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name=run_name,
        group="Derivation",
)

df_players = pd.read_csv('datasets/rotowire/player_labels.csv')[['Player Name', 'Game ID']].head(args.size)
df_reports = pd.read_csv(f'datasets/rotowire/reports_table.csv').rename(columns={'Game_ID': 'Game ID'})
df = pd.merge(df_players, df_reports, on='Game ID')

reports = pz.MemoryDataset(id="rotowire_reports_players", vals=df)
reports = reports.sem_add_columns(
    cols=[
        {
            "name": "points",
            "type": int,
            "desc": "The points scored by the player specified by the `Player Name` field in the game described by the `Report` field.",
        },
    ],
    depends_on=["Report", "Player Name"],
)

config = pz.QueryProcessorConfig(
    available_models=[model],
    timeout=50000,
)

output = reports.run(config=config)
output_df = output.to_df()

if args.provider == 'ollama':
    output_file = f"evaluation/derivation/Q1/results/palimpzest_Q1_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    output_file = f"evaluation/derivation/Q1/results/palimpzest_Q1_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

output_df.to_csv(output_file)

with open('statistics/derivation/Q1.log', 'a') as file:
    file.write(f"System: Palimpzest\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Execution Time: {output.execution_stats.total_execution_time:.2f}\n")

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=output_df),
        "execution_time": output.execution_stats.total_execution_time,
        # "total_tokens": output.execution_stats.total_tokens
    })

    wandb.finish()
