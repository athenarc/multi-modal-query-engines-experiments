import palimpzest as pz
from palimpzest.constants import Model
import pandas as pd
from dotenv import load_dotenv
import wandb

load_dotenv()

wandb.init(
    project="semantic_operations",
    name="palimpzest_Q9_gemma3_12b_ollama",
    group="semantic selection",
)

reports = pz.TextFileDataset(id="player_names", path="development/datasets/rotowire/player_names/")

reports = reports.sem_filter("The player is from America.")

config = pz.QueryProcessorConfig(
    available_models=[Model.OLLAMA_GEMMA_3_12B],
)

output = reports.run(config=config)
output_df = output.to_df()

print(output_df)

wandb.log({
    "result_table": wandb.Table(dataframe=output_df),
    "execution_time": output.execution_stats.total_execution_time,
    "total_tokens": output.execution_stats.total_tokens
})

wandb.finish()
