import pandas as pd
import lotus
from lotus.models import LM
import wandb
import time

wandb.init(
    project="semantic_operations",
    name="lotus_Q12_join_default_gemma3_12b_ollama",
    group="semantic join",
)

df_movies = pd.read_csv("development/datasets/movies_dataset/movies.csv").head(10)[['title']]
df_directors = pd.read_csv("development/datasets/movies_dataset/directors.csv")[['director_name']]

lm = LM(model="ollama/gemma3:12b")
lotus.settings.configure(lm=lm)

instruction = "The movie {title:left} is directed by {director_name:right}."
start = time.time()
df = df_movies.sem_join(df_directors, instruction)
exec_time = time.time() - start

# print(df)
# print(exec_time)

wandb.log({
    "result_table": wandb.Table(dataframe=df),
    "execution_time": exec_time
})

wandb.finish()