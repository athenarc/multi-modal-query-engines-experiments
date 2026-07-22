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
 
Each engine has its own adapter under `experiment_runner/systems/`, implementing a shared interface (`base.py`), so the same 66 queries are dispatched, executed, and scored consistently across all three rather than each engine getting its own bespoke harness.

## Structure

```bash
.
├── datasets/               # NBA (SportSett + NBA Players) and Rotten Tomatoes tables
├── experiment_runner/       
│   ├── systems/            # one adapter per engine (lotus, palimpzest, blendsql) + shared base
│   ├── configs/            # queries.yaml (the 66 queries) + run_config.yaml (run settings)
│   ├── evaluation.py       # accuracy / F1 / LLM-as-judge scoring
│   └── runner.py           # entry point
├── utilities/              # dataset preprocessing & analysis helpers
├── LICENSE
└── README.md
```

