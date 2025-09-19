import pandas as pd

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMMap, LLMQA, LLMJoin
import time
import wandb

wandb.init(
    project="semantic_operations",
    name="blendsql_Q1_llama3_1_8b_instruct_transformers",
    group="semantic projection",
)

# Load reports dataset
reports_table = pd.read_csv('development/datasets/rotowire/reports_table.csv').head(100)
reports = {
    "Reports" : pd.DataFrame(reports_table)
}

# Prepare our BlendSQL connection
bsql = BlendSQL(
    db=reports,
    model=TransformersLLM(
        "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        config={"device_map": "auto"},
        caching=False,
    ),
    # model=LiteLLM("ollama/gemma3:1b"),
    # model=LiteLLM(
    #     "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
    #     config={"api_base": "http://localhost:5001/v1"},
    # ),
    # model=LiteLLM("ollama/llama3.3:70b", config={"timeout": 50000}),
    # model=LiteLLM("ollama/qwen3:30b-a3b-instruct-2507-q4_K_M", config={"timeout": 50000}),
    verbose=True,
    ingredients={LLMMap},
)

exec_times = []
start = time.time()
smoothie = bsql.execute(
   """
    SELECT Game_ID, Reports.Report, {{
        LLMMAP(
            'Return a list of strings with player names that did play in the game described by the given report. Please ignore the players who are mentioned and did not play but returned them all.',
            Reports.Report
        )
    }}
    FROM Reports
    """,
    infer_gen_constraints=True,
)

exec_times.append(time.time()-start)


df = smoothie.df
df['player_name'] = df['_col_2'].str.replace('[', '').replace(']', '').str.split(",")
df_exploded = df.explode('player_name', ignore_index=True)
df = df_exploded.copy().drop(columns=['_col_2'])

# Points
reports = {
    "Reports" : df.copy()
}
bsql = BlendSQL(
    db=reports,
    model=TransformersLLM(
        "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        config={"device_map": "auto"},
        caching=False,
    ),
    # model=LiteLLM("ollama/gemma3:1b"),
    # model=LiteLLM("ollama/qwen3:30b-a3b-instruct-2507-q4_K_M", config={"timeout": 50000}),

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
        'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, player_name, {{LLMMap('How many points did the player have in the game? Return -1 if there are no mentions for points.', context, return_type='int')}} AS points
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

# Assists
reports = { 'Reports': smoothie.df }
bsql = BlendSQL(
    db=reports,
    model=TransformersLLM(
        "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        config={"device_map": "auto"},
        caching=False,
    ),
    # model=LiteLLM("ollama/gemma3:1b"),
    # model=LiteLLM("ollama/qwen3:30b-a3b-instruct-2507-q4_K_M", config={"timeout": 50000}),

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
        'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, player_name, points, {{LLMMap('How many assists did the player have in the game? Return -1 if there are no mentions for assists.', context, return_type='int')}} AS assists
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

# Total Rebounds
reports = {'Reports': smoothie.df }
bsql = BlendSQL(
    db=reports,
    model=TransformersLLM(
        "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        config={"device_map": "auto"},
        caching=False,
    ),
    # model=LiteLLM("ollama/gemma3:1b"),
    # model=LiteLLM("ollama/qwen3:30b-a3b-instruct-2507-q4_K_M", config={"timeout": 50000}),

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
    'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
    FROM Reports
    ) SELECT Game_ID, Report, player_name, points, assists, {{LLMMap('How many total rebounds did the player have in the game? Return -1 if there are no mentions for total rebounds.', context, return_type='int')}} AS total_rebounds
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

# Steals
reports = {'Reports': smoothie.df }
bsql = BlendSQL(
    db=reports,
    model=TransformersLLM(
        "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        config={"device_map": "auto"},
        caching=False,
    ),
    # model=LiteLLM("ollama/gemma3:1b"),
    # model=LiteLLM("ollama/qwen3:30b-a3b-instruct-2507-q4_K_M", config={"timeout": 50000}),

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
        'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, player_name, points, assists, total_rebounds, {{LLMMap('How many steals did the player have in the game? Return -1 if there are no mentions for steals.', context, return_type='int')}} AS steals
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)  

# Blocks
reports = {'Reports': smoothie.df }
bsql = BlendSQL(
    db=reports,
    model=TransformersLLM(
        "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        config={"device_map": "auto"},
        caching=False,
    ),
    # model=LiteLLM("ollama/gemma3:1b"),
    # model=LiteLLM("ollama/qwen3:30b-a3b-instruct-2507-q4_K_M", config={"timeout": 50000}),

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
        'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, player_name, points, assists, total_rebounds, steals, {{LLMMap('How many blocks did the player have in the game? Return -1 if there are no mentions for blocks.', context, return_type='int')}} AS blocks
    FROM joined_context
    """,
    
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

print(smoothie.df)

# # Defensive Rebounds
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, {{LLMMap('How many defensive rebounds did the player have in the game?', context, return_type='int')}} AS defensive_rebounds
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Offensive Rebounds
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, {{LLMMap('How many offensive rebounds did the player have in the game?', context, return_type='int')}} AS offensive_rebounds
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Personal Fouls
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, {{LLMMap('How many personal fouls did the player have in the game?', context, return_type='int')}} AS personal_fouls
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Turnovers
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, {{LLMMap('How many turnovers did the player have in the game?', context, return_type='int')}} AS turnovers
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Field Goals Made
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, {{LLMMap('How many field goals did the player make in the game?', context, return_type='int')}} AS field_goals_made
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Field Goals Attempted
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, {{LLMMap('How many field goals did the player attempt in the game?', context, return_type='int')}} AS field_goals_attempted
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Field Goals Percentage
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, {{LLMMap('What was the field goal percentage of player in the game?', context, return_type='int')}} AS field_goals_percentage
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Free Throws Made
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, {{LLMMap('How many free throws did the player make in the game?', context, return_type='int')}} AS free_throws_made
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Free Throws Attempted
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, free_throws_made, {{LLMMap('How many free throws did the player attempt in the game?', context, return_type='int')}} AS free_throws_attempted
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Free Throws Percentage
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, free_throws_made, free_throws_attempted, {{LLMMap('What was the free throw percentage of player in the game?', context, return_type='int')}} AS free_throw_percentage
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)



# # 3-pointers made
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, free_throws_made, free_throws_attempted, free_throw_percentage, {{LLMMap('How many 3-pointers did the player make in the game?', context, return_type='int')}} AS three_pointers_made
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)


# # 3-pointers attempted
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, free_throws_made, free_throws_attempted, free_throw_percentage, three_pointers_made, {{LLMMap('How many 3-pointers did the player attempt in the game?', context, return_type='int')}} AS three_pointers_attempted
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # 3-pointers Percentage
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, free_throws_made, free_throws_attempted, free_throw_percentage, three_pointers_made, three_pointers_attempted, {{LLMMap('What was the 3-pointers percentage for player in the game?', context, return_type='int')}} AS three_pointers_percentage
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Minutes Played
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, free_throws_made, free_throws_attempted, free_throw_percentage, three_pointers_made, three_pointers_attempted, three_pointers_percentage, {{LLMMap('How many minutes did the player play in the game?', context, return_type='int')}} AS minutes_played
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

wandb.log({
    "result_table": wandb.Table(dataframe=smoothie.df.fillna(-1)),
    "execution_time": sum(exec_times)
})

wandb.finish()

