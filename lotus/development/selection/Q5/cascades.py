import pandas as pd
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs
import time
import wandb

# wandb.init(
#     project="semantic_operations",
#     name="lotus_Q5_filter_cascades_gemma3_12B_ollama",
#     group="semantic selection",
# )

lm = LM(model="ollama/gemma3:12b")
helper_lm = LM(model="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct", api_base="http://localhost:5001/v1", api_key="dummy")

lotus.settings.configure(lm=lm, helper_lm=helper_lm)
df_reviews = pd.read_csv("development/datasets/imdb_dataset.csv").head(100)

user_instruction = "{review} is positive"

cascade_args = CascadeArgs(recall_target=0.9, precision_target=0.9, sampling_percentage=0.2, failure_probability=0.1)

start = time.time()
df = df_reviews.sem_filter(user_instruction, cascade_args=cascade_args)
exec_time = time.time() - start

# print(df)
# print(exec_time)

# wandb.log({
#     "result_table": wandb.Table(dataframe=df),
#     "execution_time": exec_time
# })
# wandb.finish()