import pandas as pd

from blendsql import BlendSQL
from blendsql.models import TransformersLLM
from blendsql.ingredients import LLMMap, LLMQA, LLMJoin
from time import time
import wandb

wandb.init(
    project="semantic_operations",
    name="blendsql_RoT",
    group="semantic projection",
)

# Load reports
reports = pd.read_csv('reports_table.csv').head(2)
reports = {
    "Reports" : pd.DataFrame(reports['Report'])
}

# Prepare our BlendSQL connection
bsql = BlendSQL(
    db=reports,
    model=TransformersLLM(
        "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        config={"device_map": "auto"},
        caching=False,
    ),
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()
smoothie = bsql.execute(
   """
    SELECT Reports.Report, {{
        LLMMAP(
            'Return the name of the player that scored the most points.',
            Reports.Report,
        )
    }}
    FROM Reports
    """,
    infer_gen_constraints=True,
)
exec_time = time.time() - start

wandb.log({
    "result_table": wandb.Table(dataframe=smoothie.df),
    "execution_time": exec_time
})

wandb.finish()
