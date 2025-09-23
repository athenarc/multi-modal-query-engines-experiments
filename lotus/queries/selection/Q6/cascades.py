import pandas as pd
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs
import time
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
args = parser.parse_args()

if args.wandb:
    wandb.init(
        project="semantic_operations",
        name="lotus_Q6_filter_cascades_gemma3_12B_ollama_llama8b_vllm",
        group="semantic selection",
    )

lm = LM(model="ollama/gemma3:12b")
helper_lm = LM(model="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct", api_base="http://localhost:5001/v1", api_key="dummy")

lotus.settings.configure(lm=lm, helper_lm=helper_lm)
df_players = pd.read_csv("datasets/rotowire/reports_with_players_100.csv")
df_players = df_players[df_players["Game ID"] < 14] # Keep total of ~100 entries

user_instruction = "{Player Name} had 17 points in game {Report}."

cascade_args = CascadeArgs(recall_target=0.9, precision_target=0.9, sampling_percentage=0.2, failure_probability=0.1)

start = time.time()
df = df_players.sem_filter(user_instruction, cascade_args=cascade_args)
exec_time = time.time() - start

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", df)
    print("Execution time: ", exec_time)