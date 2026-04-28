import pandas as pd
import time
import wandb
import argparse

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMQA

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=1000, const=1000, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"blendsql_Q16_aggregation_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name=run_name,
        group="Aggregation",
    )

df_reviews_sample = pd.read_csv("datasets/imdb_reviews/imdb_reviews.csv").sample(args.size)
df_reviews_sample.to_csv(f"datasets/imdb_reviews/imdb_reviews_sampled_{args.size}.csv", index=False)
df_reviews = df_reviews_sample[['review']]

if args.provider == 'ollama':
    model=LiteLLM(args.provider + '/' + args.model, config={"timeout": 50000}, caching=False)
elif args.provider == 'vllm':
     model = LiteLLM("hosted_vllm/" + args.model, 
                    config={"api_base": "http://localhost:5001/v1", "timeout": 50000, "cache": False}, 
                    caching=False)

db = {
    "Reviews": pd.DataFrame(df_reviews)
}

bsql = BlendSQL(
    db=db,
    model=model,
    # model=TransformersLLM(
    #     "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    #     config={"device_map": "auto"},
    #     caching=False,
    # ),
    ingredients={LLMQA}
)

start = time.time()
smoothie = bsql.execute(
    """
        SELECT {{
            LLMQA(
                'Count positive reviews. Return **only** a number.',
                context=Reviews.review,
            )
        }} AS Answer
    """,
    infer_gen_constraints=True,
)

exec_time = time.time()-start

with open('statistics/aggregation/Q16.log', 'a') as file:
    file.write(f"System: BlendSQL (LLMQA)\n")
    file.write(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Execution Time: {exec_time:.2f}\n")
    file.write(f"Result: {smoothie.df['Answer'].iloc[0]}" + "\n")

if args.wandb:
    wandb.log({
        "result": wandb.Table(dataframe=smoothie.df),
        "execution_time": exec_time
    })
    wandb.finish()