import pandas as pd
import time
import wandb
import argparse

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMJoin

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=10, const=10, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"blendsql_Q12_join_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic join",
    )

# Load reports dataset
movies_df = pd.read_csv('datasets/movies_directors/movies.csv').head(args.size)[['title']]

directors_df = pd.read_csv('datasets/movies_directors/directors.csv')[['director_name']]

db = {
    "Movies": pd.DataFrame(movies_df),
    "Directors": pd.DataFrame(directors_df)
}

if args.provider == 'ollama':
    model = LiteLLM(args.provider + '/' + args.model, 
                    config={"timeout" : 50000, "cache": False},
                    caching=False)
elif args.provider == 'vllm':
    model = LiteLLM("hosted_vllm/" + args.model, 
                    config={"api_base": "http://localhost:5001/v1", "timeout": 50000, "cache": False}, 
                    caching=False)
    
bsql = BlendSQL(
    db=db,
    model=model,
    ingredients={LLMJoin},
)

start = time.time()
smoothie = bsql.execute(
    """
        SELECT *
        FROM Movies m
        JOIN Directors d ON {{
            LLMJoin(
                m.title,
                d.director_name,
                join_criteria='The movie is directed from the director.',
            )
        }} 
    """,
    infer_gen_constraints=True,
)

exec_time = time.time()-start

if args.wandb:
    smoothie.df.to_csv(f"evaluation/join/Q12/results/blendsql_Q12_join_{args.model.replace('/', ':')}_{args.provider}_{args.size}.csv")

    wandb.log({
        "result_table": wandb.Table(dataframe=smoothie.df),
        "execution_time": exec_time
    })

    wandb.finish()
else:


    print("Result:\n\n", pd.DataFrame(smoothie.df))
    print("Execution time: ", exec_time)