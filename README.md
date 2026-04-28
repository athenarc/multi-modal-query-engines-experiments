# Towards Semantic Query Engines: Finding a Common Ground [Experiment, Analysis & Benchmark]
This repository contains the official code for the paper "Towards Semantic Query Engines: Finding a Common Ground".
The current version aims to experiment with three LLM-based Semantic Query Engines (SQEs) -Lotus, Palimpzest, BlendSQL- and one SLM-based engine. We evaluate these systems over a unified setting of 18 newly designed queries.
We manually scrapted all the scripts for creating the corresponding queries for each LLM-based system. For the ELEET system, we include only the evaluation scripts, as we utilized its official implementation for the query execution.

<p align="center">
  <img src="Queries.png" title="Query Set">
</p>

## Abstract
*Deep learning has enabled semantic operations opening the door to querying heterogeneous data far beyond the limits of classical relational systems. A growing set of Semantic Query Engines (SQEs) integrates these capabilities, but the landscape remains fragmented, lacking shared operator definitions, architectural patterns, and comparable evaluation. This paper provides the first systematic analysis of SQEs across semantic operator classes, query capabilities, and system architectures. We introduce taxonomies for semantic operators and SQE query expressiveness, and develop operator comparison graphs to characterize operator substitutability across systems. Our empirical study across representative SQEs reveals fundamental challenges and gaps laying the groundwork for more coherent and interoperable next-generation SQEs.*

## Structure
All directories that are linked to a system are structured in a identical manner:

```bash
system/
├── environment.yml
└── queries/
    └── <operation_class>/           # derivation, selection, join or aggregation
        ├── <query_name>/            # One of (Q1, Q2, ..., Q18)
        └── <execution_script>.sh    # Bash script for executing the query with multiple configurations
  ```

## Reproducability
The enviroment for each system is set up using Conda. Each system has its own `environment.yml`. To set up a system:

```
# Example for Lotus
cd lotus
conda env create -f environment.yml
conda activate lotus
```

Each directory corresponding to a semantic operation class includes a bash script as shown in the structure, which receives the LLM provider (vLLM or Ollama) along with the set of queries that will be executed (strict order), and executes both query and evaluation scripts with different input sizes. As of now, we have linked vLLM to the LLM that we used to report our measurements (i.e. `Llama-3.3-70B-quantized.w8a8` from RedHatAI) and Ollama with the `gemma3:12b`, either with its default context window length or modified to 128.000 tokens (see the `.Modfile`). Note that all bash scripts must be executed from the root of the project.

Usage Example:
```bash
./lotus/queries/derivation/derivation_runs.sh vllm 1 3 6
```

This command executes the Lotus implementation for queries Q1, Q3, and Q6 using `RedHatAI/Llama-3.3-70b-quantized.w8a8` as the underlying model. Subsequently, it evaluates the outputs and saves logs to the statistics directory, recording both execution time and result quality.

