import pandas as pd
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs
import time
import wandb

wandb.init(
    project="semantic_operations",
    name="lotus_Q5_filter_default_gemma3_12B_ollama",
    group="semantic selection",
)

lm = LM(model="ollama/gemma3:12b")

lotus.settings.configure(lm=lm)
df_reviews = pd.read_csv("/data/hdd1/users/jzerv/reproduced-systems/Lotus/development/imdb_dataset.csv").head(100)
df_reviews = pd.DataFrame(df_reviews['review'])

user_instruction = "{review} is positive"

start = time.time()
df = df_reviews.sem_filter(user_instruction)
exec_time = time.time() - start

print(df)
print(exec_time)

wandb.log({
    "result_table": wandb.Table(dataframe=df),
    "execution_time": exec_time
})
wandb.finish()