import pandas as pd

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMJoin
import time
import wandb

wandb.init(
    project="semantic_operations",
    name="blendsql_q12_gemma3_12b_ollama",
    group="semantic join",
)

# Load reports dataset
movies_df = pd.read_csv('development/datasets/movies_dataset/movies.csv').head(10)[['title']]

directors_df = pd.read_csv('development/datasets/movies_dataset/directors.csv')[['director_name']]

db = {
    "Movies": pd.DataFrame(movies_df),
    "Directors": pd.DataFrame(directors_df)
}

bsql = BlendSQL(
    db=db,
    model=LiteLLM("ollama/gemma3:12b", config={"timeout": 50000}),
    ingredients={LLMJoin}
)

start = time.time()
smoothie = bsql.execute(
    """
        SELECT *
        FROM Movies m
        JOIN Directors d ON {{
            LLMJoin(
                'm.title',
                'd.director_name',
                question='The movie is directed from the director.',
            )
        }} 
    """,
    infer_gen_constraints=True,
)

exec_time = time.time()-start

# print(df)
# print(exec_time)

wandb.log({
    "result_table": wandb.Table(dataframe=smoothie.df),
    "execution_time": exec_time
})

wandb.finish()


