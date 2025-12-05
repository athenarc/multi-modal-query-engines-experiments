import pandas as pd
import time
import wandb
import argparse
from datetime import datetime

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMMap, LLMQA, LLMJoin

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"blendsql_Q2_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name=run_name,
        group="Derivation",
)

# Load reports
reports = pd.read_csv('datasets/rotowire/player_evidence_mine.csv').dropna(subset=['birth_place']).head(args.size)
reports.rename(columns={"Player Name": "player_name"}, inplace=True)
players = {
    "Players" : pd.DataFrame(reports['player_name'])
}

if args.provider == 'ollama':
    model = LiteLLM(args.provider + '/' + args.model, 
                    config={"timeout" : 50000, "cache": False},
                    caching=False)
elif args.provider == 'vllm':
    model = LiteLLM("hosted_vllm/" + args.model, 
                    config={"api_base": "http://localhost:5001/v1", "timeout": 50000, "cache": False}, 
                    caching=False)
elif args.provider == 'transformers':
    model = TransformersLLM(
        "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        config={"device_map": "auto"},
        caching=False,
    )


# Prepare our BlendSQL connection
bsql = BlendSQL(
    db=players,
    model=model,
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()

smoothie = bsql.execute(
   """
    SELECT Players.player_name, {{
        LLMMAP(
            'What is the birthplace of the player? Please return a string containing **only** the birth place (no think).',
            return_type='str',
            Players.player_name,
        )
    }}
    FROM Players
    """,
    infer_gen_constraints=True,
)

exec_time = time.time() - start

smoothie.df.to_csv(f"evaluation/derivation/Q2/results/blendsql_Q2_map_{args.model.replace('/', '_').replace(':', '_')}_{args.provider}_{args.size}.csv")

with open('statistics/derivation/Q2.log', 'a') as file:
    file.write(f"System: BlendSQL\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Execution Time: {exec_time:.2f}\n")
    file.write("Total LLM calls: " + str(args.size) + "\n")

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=smoothie.df),
        "execution_time": exec_time
    })
    wandb.finish()