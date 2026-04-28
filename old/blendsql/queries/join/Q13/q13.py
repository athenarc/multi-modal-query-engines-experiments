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
    run_name = f"blendsql_Q13_join_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name=run_name,
        group="Join",
    )

# Load reports dataset
df_text = pd.read_csv("datasets/banking_data/banking_data_test.csv")[['text']].head(args.size)
df_categories = pd.read_csv("datasets/banking_data/categories.csv").head(63)

db = {
    "Texts": pd.DataFrame(df_text),
    "Categories": pd.DataFrame(df_categories)
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
        FROM Texts t
        JOIN Categories c ON {{
            LLMJoin(
                t.text,
                c.category,
                join_criteria='The online banking query/text maps the category-intent.',
            )
        }} 
    """,
    infer_gen_constraints=True,
)

exec_time = time.time()-start

output_file = f"evaluation/join/Q13/results/blendsql_Q13_join_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
smoothie.df.to_csv(output_file)
with open('statistics/join/Q13.log', 'a') as file:
    file.write(f"System: BlendSQL (LLMJoin)\n")
    file.write(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Execution Time: {exec_time:.2f}\n")

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=smoothie.df),
        "execution_time": exec_time
    })

    wandb.finish()