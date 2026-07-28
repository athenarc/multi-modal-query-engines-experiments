# Towards Semantic Query Engines: Finding a Common Ground

Official experiment code for the under-submission paper **"Towards Semantic Query Engines: Finding a Common Ground"** *(Experiments & Analysis)*.

## Abstract

*Deep learning has enabled **semantic operations** — LLM- and embedding-driven classification, extraction, ranking, joining, and aggregation — that let query engines go beyond classical relational primitives. A growing family of **Semantic Query Engines (SQEs)** integrates these capabilities, but the landscape is fragmented: no shared operator definitions, no common architectural patterns, and no comparable evaluation. This paper provides the first systematic analysis of SQEs across semantic operator classes, query capabilities, and system architectures, and this repository contains the code used to run that empirical study.*

---

This repository runs the paper's full **66-query benchmark** across three representative Semantic Query Engines, evaluated under a single unified runner:

| System | Version |
|---|---|
| **[Lotus](https://github.com/lotus-data/lotus)** | v1.2.4 |
| **[Palimpzest](https://github.com/mitdbg/palimpzest)** | v1.5.3 |
| **[BlendSQL](https://github.com/parkervg/blendsql)** | v0.1.26 |

Each SQE has its own adapter under `experiment_runner/systems/`, implementing a shared interface (`base.py`), so the same 66 queries are dispatched, executed, and scored consistently across all three engines.

## Core Structure

```bash
.
├── datasets/                  # NBA (SportSett + NBA Players Info) and Rotten Tomatoes tables
├── experiment_runner/
│   ├── systems/                # one adapter per engine (lotus, palimpzest, blendsql) + shared base
│   ├── configs/
│   │   ├── queries.yaml        # 66 query definitions (nlq, class, task, input)
│   │   └── run_config.yaml     # engine, model, track, and filter settings
│   ├── evaluation.py           # quality score per semantic class
│   └── runner.py               # entry point
├── utilities/                  # dataset preprocessing & analysis helpers
└── README.md
```

`queries.yaml` holds the 66 query definitions (nlq, class, task, input) and `run_config.yaml` controls which engine(s), model(s), and track (scalability or quality) we target.

<!-- Query catalog table (semantic class breakdown, counts, example NLQs) to be added before camera-ready. -->

## Reproducibility

For each system, you have to install the corresponding python package (prefer to create a virtual environment for each SQE). Refer to the corresponding repository for details about installing an SQE.

```bash
pip install [lotus-ai (python 3.10), blendsql (python >=3.10) or palimpzest (python >=3.12)]
```

Additionally, you need either vLLM and Ollama installations, installed alongside with the LLMs you aim to use.

## Running an Experiment

In order to run an experiment, first you need to activate the corresponding environment. Afterwards, uncomment the lines of code associated only with the system you want to test in file `experiment_runner/systems/__init__.py`. This is necessary because each SQE lives in its own virtual environment — importing all three adapters unconditionally would require every package installed at once. For example, if you want to run queries using Lotus, you need to keep only the following two lines of code uncommented:

```python
from .lotus import LotusSystem
...
   "lotus": LotusSystem,
```

- `experiment_runner/configs/queries.yaml` contains the set of all queries, over both NBA and Rotten Tomatoes datasets.
- `experiment_runner/configs/run_config.yaml` is responsible for setting up all the parameters that are used in the entry point `experiment_runner/runner.py`. You have to set the system, LLM, input sizes, experimental track, wandb report parameters. Finally, you can filter the queries by restricting the execution over a specific semantic class, task or specific queries.

Once configured, run the entry point from the `experiment_runner` directory:

```bash
cd experiment_runner
python runner.py
```

Results (per-query scores and run metadata) are written locally and, if enabled in `run_config.yaml`, logged to Weights & Biases.
