import palimpzest as pz
from palimpzest.constants import Model
import pandas as pd
from dotenv import load_dotenv
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
args = parser.parse_args()

load_dotenv()

if args.wandb:
    wandb.init(
        project="semantic_operations",
        name="palimpzest_Q10_gemma3_12b_ollama",
        group="semantic selection",
    )

reports = pz.TextFileDataset(id="team_names", path="datasets/rotowire/team_names/")

reports = reports.sem_filter("The team was founded before 1970.")

config = pz.QueryProcessorConfig(
    available_models=[Model.OLLAMA_GEMMA_3_12B],
)

output = reports.run(config=config)
output_df = output.to_df()

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=output_df),
        "execution_time": output.execution_stats.total_execution_time,
        "total_tokens": output.execution_stats.total_tokens
    })

    wandb.finish()
else:
    print("Result:\n\n", output_df)
    print("Execution time: ", output.executions_stats.total_execution_time)
