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
parser.add_argument("-s", "--size", nargs='?', default=50, const=50, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"blendsql_Q4_{args.model.replace(':', '_').replace('/', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name=run_name,
        group="Derivation",
    )

# Load reports dataset
df_reports = pd.read_csv("datasets/rotowire/reports_table.csv")
df_player_names = pd.read_csv('datasets/rotowire/player_labels.csv')[['Player Name', 'Game ID']].rename(columns={'Game ID': 'Game_ID'})
df_init = pd.merge(df_player_names, df_reports, on='Game_ID').rename(columns={'Player Name': 'player_name'}).head(args.size)

reports = {
    "Reports" : pd.DataFrame(df_init)
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
    db=reports,
    model=model,
    verbose=True,
    ingredients={LLMMap},
)

exec_times = []

# Assists
reports = { 'Reports': df_init }
bsql = BlendSQL(
    db=reports,
    model=model,
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()
smoothie = bsql.execute(
   """
    WITH joined_context AS (
        SELECT *,
        'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, player_name, {{LLMMap('How many assists did the player have in the game (-1 if there are no mentions)? Return **only** an integer.', context, return_type='int')}} AS assists
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

# Total Rebounds
reports = {'Reports': smoothie.df }
bsql = BlendSQL(
    db=reports,
    model=model,
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()
smoothie = bsql.execute(
   """
    WITH joined_context AS (
    SELECT *,
    'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
    FROM Reports
    ) SELECT Game_ID, Report, player_name, assists, {{LLMMap('How many total rebounds did the player have in the game (-1 if there are no mentions)? Return **only** an integer.', context, return_type='int')}} AS total_rebounds
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

# Steals
reports = {'Reports': smoothie.df }
bsql = BlendSQL(
    db=reports,
    model=model,
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()
smoothie = bsql.execute(
   """
    WITH joined_context AS (
        SELECT *,
        'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, player_name, assists, total_rebounds, {{LLMMap('How many steals did the player have in the game (-1 if there are no mentions)? Return **only** an integer.', context, return_type='int')}} AS steals
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)  

# Blocks
reports = {'Reports': smoothie.df }
bsql = BlendSQL(
    db=reports,
    model=model,
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()
smoothie = bsql.execute(
   """
    WITH joined_context AS (
        SELECT *,
        'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, player_name, assists, total_rebounds, steals, {{LLMMap('How many blocks did the player have in the game (-1 if there are no mentions)? Return **only** an integer.', context, return_type='int')}} AS blocks
    FROM joined_context
    """,
    
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)
exec_time = sum(exec_times)

print("saving")
smoothie.df.to_csv(f"evaluation/derivation/Q4/results/blendsql_Q4_{args.model.replace('/', '_').replace(':', '_')}_{args.provider}_{args.size}.csv")

total_LLM_calls = args.size * 4

with open('statistics/derivation/Q4.log', 'a') as file:
    file.write(f"System: BlendSQL\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Execution Time: {exec_time:.2f}\n")
    file.write("Total LLM calls: " + str(total_LLM_calls) + "\n")

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=smoothie.df.fillna(-1)),
        "execution_time": exec_time,
        "total_LLM_calls": total_LLM_calls,
    })
    wandb.finish()