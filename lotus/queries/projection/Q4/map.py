import pandas as pd
import lotus
from lotus.models import LM
import time
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
args = parser.parse_args()

if args.wandb:
    wandb.init(
        project="semantic_operations",
        name="lotus_map_Q4_llama3.3_70b_ollama",
        group="semantic projection",
    )

lm = LM(model="ollama/llama3.3:70b")
lotus.settings.configure(lm=lm)

df_reports = pd.read_csv("datasets/rotowire/player_evidence_mine.csv")
df = df_reports['Player Name'].to_frame(name='Player Name')

start = time.time()

user_instruction = "What is the birthdate of player {Player Name}? Please return only the birthdate in format DD/MM/YYYY."
df_birthdate = df.sem_map(user_instruction)
df['birthdate'] = df_birthdate['_map']

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