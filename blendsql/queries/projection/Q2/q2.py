import pandas as pd
import time
import wandb
import argparse

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMMap, LLMQA, LLMJoin

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
args = parser.parse_args()

if args.wandb:
    wandb.init(
        project="semantic_operations",
        name="blendsql_NER_TE_Q2_gemma3_12b_ollama",
        group="semantic projection",

)

# Load reports dataset
reports_table = pd.read_csv('datasets/rotowire/reports_table.csv').head(100)
reports = {
    "Reports" : pd.DataFrame(reports_table)
}

# Prepare our BlendSQL connection
bsql = BlendSQL(
    db=reports,
    # model=TransformersLLM(
    #     "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    #     config={"device_map": "auto"},
    #     caching=False,
    # ),
    # model=LiteLLM(
    #     "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
    #     config={"api_base": "http://localhost:5001/v1"},
    # ),
    model=LiteLLM("ollama/gemma3:12b", config={"timeout": 50000}),
    verbose=True,
    ingredients={LLMMap},
)

exec_times = []
start = time.time()

smoothie = bsql.execute(
   """
    SELECT Reports.Game_ID, Reports.Report, {{
        LLMMAP(
            'Return a list of strings with team names that did play in the game. Please ignore the teams who are mentioned and did not play.',
            Reports.Report,
        )
    }}
    FROM Reports
    """,
    infer_gen_constraints=True,
)

exec_times.append(time.time()-start)

print(smoothie.df)
df = smoothie.df
df['team_name'] = df['_col_2'].str.split(",")
df_exploded = df.explode('team_name', ignore_index=True)
df = df_exploded.copy().drop(columns=['_col_2'])

# Points
reports = {'Reports': df.copy() }
bsql = BlendSQL(
    db=reports,
    # model=TransformersLLM(
    #     "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    #     config={"device_map": "auto"},
    #     caching=False,
    # ),
    model=LiteLLM("ollama/gemma3:12b", config={"timeout": 50000}),
    # model=LiteLLM(
    #     "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
    #     config={"api_base": "http://localhost:5001/v1"},
    # ),
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()
smoothie = bsql.execute(
   """
    WITH joined_context AS (
        SELECT *,
        'Team: ' || CAST(team_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Reports.Game_ID, Report, team_name, {{LLMMap('How many Wins has the team?', context, return_type='int')}} AS wins
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

# Assists
reports = { 'Reports': smoothie.df }
bsql = BlendSQL(
    db=reports,
    # model=TransformersLLM(
    #     "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    #     config={"device_map": "auto"},
    #     caching=False,
    # ),
    model=LiteLLM("ollama/gemma3:12b", config={"timeout": 50000}),
    # model=LiteLLM(
    #     "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
    #     config={"api_base": "http://localhost:5001/v1"},
    # ),
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()
smoothie = bsql.execute(
   """
    WITH joined_context AS (
        SELECT *,
        'Team: ' || CAST(team_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Reports.Game_ID, Report, team_name, wins, {{LLMMap('How many Losses has the team?', context, return_type='int')}} AS losses
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

# Total Rebounds
reports = {'Reports': smoothie.df }
bsql = BlendSQL(
    db=reports,
    # model=TransformersLLM(
    #     "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    #     config={"device_map": "auto"},
    #     caching=False,
    # ),
    model=LiteLLM("ollama/gemma3:12b", config={"timeout": 50000}),
    # model=LiteLLM(
    #     "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
    #     config={"api_base": "http://localhost:5001/v1"},
    # ),
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()
smoothie = bsql.execute(
   """
    WITH joined_context AS (
    SELECT *,
    'Team: ' || CAST(team_name AS VARCHAR) || '\nReport: ' || Report AS context
    FROM Reports
    ) SELECT Reports.Game_ID, Report, team_name, wins, losses, {{LLMMap('What is the number of total points the team scored?', context, return_type='int')}} AS total_points
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=smoothie.df),
        "execution_time": sum(exec_times)
    })

    wandb.finish()
else:
    print("Result:\n\n", smoothie.df)
    print("Execution time: ", sum(exec_times))


