import yaml
import itertools
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional
from systems import get_system

class Query(BaseModel):
    id: str
    class_name: str
    task_name: str
    nlq: str
    table: str
    cols: List[str]
    lotus_query: Optional[str] = None

class ExperimentRunner:
    def __init__(self, run_config_path: str, queries_path: str):
        self.run_config = self._load_yaml(run_config_path)
        self.all_queries = [Query(**q) for q in self._load_yaml(queries_path)['queries']]
        self.results = []

    def _load_yaml(self, path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def filter_queries(self) -> List[Query]:
        filters = self.run_config.get('filters', {})
        target_class = filters.get('class_name')
        target_task = filters.get('task_name')

        filtered = self.all_queries
        if target_class:
            filtered = [q for q in filtered if q.class_name == target_class]
        if target_task:
            filtered = [q for q in filtered if q.task_name == target_task]
        
        return filtered

    def run(self):
        queries_to_run = self.filter_queries()

        print(f"Starting experiment: {self.run_config['experiment_name']}")
        print(f"Total Queries: {len(queries_to_run)}")

        for system_name, llm_provider in itertools.product(self.run_config['systems'], self.run_config['llm_providers']):
            # Initialize system once per LLM-Provider combination
            system_instance = get_system(system_name, llm_provider, self.run_config['model_names'][0])

            for query in queries_to_run:
                print(query.lotus_query)

                print(f"Executing Query {query.id} ({query.class_name} / {query.task_name})...")
            
                for input_size in self.run_config['input_sizes']:
                    print(f"\n--- Running: {system_name.upper()} | {self.run_config['model_names'][0]} |{llm_provider} | Size: {input_size} ---")
                    
                    try:
                        system_query = getattr(query, f"{system_name}_query", None)
                        if system_query is None:
                            raise ValueError(f"No query defined for {system_name} in query ID {query.id}")

                        output = system_instance.execute_query(query.class_name, system_query, query.table, query.cols, input_size)

                        # Log successful result
                        self.results.append({
                            "experiment": self.run_config['experiment_name'],
                            "system": system_name,
                            "llm": llm_provider,
                            "input_size": input_size,
                            "query_id": query.id,
                            "class_name": query.class_name,
                            "task_name": query.task_name,
                            "latency_sec": output.get('latency'),
                            "status": "success",
                            "error": None
                        })
                    except Exception as e:
                        print(f"Error on {query.id}: {str(e)}")
                        # Log failed result
                        self.results.append({
                             "experiment": self.run_config['experiment_name'],
                            "system": system_name,
                            "llm": llm_provider,
                            "input_size": input_size,
                            "query_id": query.id,
                            "class_name": query.class_name,
                            "task_name": query.task_name,
                            "latency_sec": None,
                            "status": "failed",
                            "error": str(e)
                        })

            self.save_results()

    def save_results(self):
        df = pd.DataFrame(self.results)
        filename = f"results_{self.run_config['experiment_name']}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ Experiments complete! Results saved to {filename}")

if __name__ == "__main__":
    runner = ExperimentRunner(
        run_config_path="configs/run_config.yaml",
        queries_path="configs/queries.yaml"
    )
    runner.run()