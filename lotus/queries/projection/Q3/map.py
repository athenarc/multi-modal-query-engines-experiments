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
        name="lotus_map_Q3_gemma3.12_ollama",
        group="semantic projection",
    )

lm = LM(model="ollama/gemma3:12b")
lotus.settings.configure(lm=lm)

df_players = pd.read_csv("datasets/rotowire/player_evidence_mine.csv")
df_players.dropna(subset=['nationality'], inplace=True)
df = df_players['Player Name'].to_frame(name='Player Name')

start = time.time()

user_instruction = "What is the nationality of player {Player Name}? Please return only the nationality."
df_nationality = df.sem_map(user_instruction)
df['nationality'] = df_nationality['_map']

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