Readme · MD
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
├── datasets/               # NBA (SportSett + NBA Players Info) and Rotten Tomatoes tables
├── experiment_runner/       
│   ├── systems/            # one adapter per engine (lotus, palimpzest, blendsql) + shared base
│   ├── configs/
        ├── queries.yaml    
        └── run_config.yaml
│   ├── evaluation.py       # quality score per semantic class
│   └── runner.py           # entry point
├── utilities/              # dataset preprocessing & analysis helpers
└── README.md
```

`queries.yaml` holds the 66 query definitions (nlq, class, task, input) and `run_config.yaml` controls which engine(s), model(s), and track (scalability or quality) we target.

<!-- TODO: ADD QUERIES TABLES -->

## Reproducability

## Running an Experiment
