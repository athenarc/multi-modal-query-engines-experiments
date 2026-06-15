import yaml
import itertools
import pandas as pd
import wandb
import os
from pydantic import BaseModel
from typing import List, Optional
from systems import get_system
from evaluation import DerivationEvaluator, SelectionEvaluator, JoinEvaluator

class Query(BaseModel):
    id: str
    class_name: str
    task_name: str
    nlq: str
    
    table: Optional[str] = None
    quality_table: Optional[str] = None
    scalability_table: Optional[str] = None
    cols: Optional[List[str]] = None
    
    # Join queries
    table_left: Optional[str] = None
    cols_left: Optional[List[str]] = None
    table_right: Optional[str] = None
    cols_right: Optional[List[str]] = None
    left_key: str = None
    right_key: str = None
    evaluation_table: str = None
    
    new_col_name: Optional[str] = None
    new_col_type: Optional[str] = str
    lotus_query: Optional[str] = None
    palimpzest_query: Optional[str] = None
    blendsql_query: Optional[str] = None
    evaluation_cols: Optional[List[str]] = None

    # Selection queries
    filtering_col: Optional[str] = None

class ExperimentRunner:
    def __init__(self, run_config_path: str, queries_path: str):
        self.run_config = self._load_yaml(run_config_path)
        self.all_queries = [Query(**q) for q in self._load_yaml(queries_path)['queries']]

    def _load_yaml(self, path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _evaluate_results(self, query: Query, predicted_df: pd.DataFrame, input_size: int, input_folder="datasets/nba/quality_exps/") -> int:
        """Evaluate predicted results and return quality metric."""        
        try:
            if query.class_name == 'derivation':
                evaluator = DerivationEvaluator(query_id=query.id, class_name=query.class_name)
                results = evaluator.evaluate(
                    predicted_df=predicted_df,
                    ground_truth_table_name= input_folder + query.table,
                    input_size=input_size,
                    evaluation_cols=query.evaluation_cols,
                    new_col_name=query.new_col_name,
                    ground_truth_col_name=query.evaluation_cols[-1],
                )

                exact_match_accuracy = results.get('exact_match_accuracy')
                similarity_accuracy = results.get('similarity_accuracy')
                
                # Return accuracy as quality for derivation tasks
                if exact_match_accuracy is not None and similarity_accuracy is not None:
                    return (float(exact_match_accuracy), float(similarity_accuracy))
                else:
                    return (-1, -1)

            if query.class_name == "selection":
                evaluator = SelectionEvaluator(query_id=query.id, class_name=query.class_name)
                results = evaluator.evaluate(
                    predicted_df=predicted_df,
                    ground_truth_table_name= input_folder + query.table,
                    input_size=input_size,
                    evaluation_cols=query.evaluation_cols,
                    filtering_col=query.filtering_col
                )
    
                accuracy = results.get('accuracy')
                recall = results.get('recall')
                precision = results.get('precision')
                f1_score = results.get('f1_score')

                if accuracy is not None and recall is not None and precision is not None and f1_score is not None:
                    return (float(accuracy), float(recall), float(precision), float(f1_score))
                else:
                    return (-1, -1, -1, -1)


            if query.class_name == "join":
                evaluator = JoinEvaluator(query_id=query.id, class_name=query.class_name)
                results = evaluator.evaluate(
                    predicted_df=predicted_df,
                    table_left_name= input_folder + query.table_left,
                    table_right_name= input_folder + query.table_right,
                    input_size=input_size,
                    evaluation_table_name=query.evaluation_table,
                    evaluation_cols=query.evaluation_cols,
                    left_key=query.left_key,
                    right_key=query.right_key
                )

                recall = results.get('recall')
                precision = results.get('precision')
                f1_score = results.get('f1_score')

                if recall is not None and precision is not None and f1_score is not None:
                    return(float(recall), float(precision), float(f1_score))
                else:
                    return(-1, -1, -1)

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return -1

    def filter_queries(self) -> List[Query]:
        filters = self.run_config.get('filters', {})
        target_class = filters.get('class_name')
        target_task = filters.get('task_name')
        target_ids = filters.get('query_ids')

        filtered = self.all_queries
        
        # Filter by Query IDs (if provided and not empty)
        if target_ids:
            filtered = [q for q in filtered if q.id in target_ids]
            
        # Filter by Class Name
        if target_class:
            filtered = [q for q in filtered if q.class_name == target_class]
            
        # Filter by Task Name
        if target_task:
            filtered = [q for q in filtered if q.task_name == target_task]
        
        return filtered

    def run(self):
        queries_to_run = self.filter_queries()

        valid_llm_configs = []
        for provider, models in self.run_config.get('llms', {}).items():
            for model_name in models:
                valid_llm_configs.append((provider, model_name))

        print(f"Starting experiment: {self.run_config['experiment_name']}")
        print(f"Total Queries after filtering: {len(queries_to_run)}")

        if not queries_to_run:
            print("No queries matched the filters. Exiting.")
            return
        
        if self.run_config['quality_exps']:
            input_folder = f"../datasets/nba/quality_exps/"
        else:
            input_folder = "../datasets/nba/scalability_exps/"

        for system_name, (llm_provider, model_name) in itertools.product(self.run_config['systems'], valid_llm_configs):
            # Initialize system once per LLM-Provider-Model combination
            system_instance = get_system(system_name, llm_provider, model_name)

            for query in queries_to_run:
                print(f"Executing Query {query.id} ({query.class_name} / {query.task_name})...")

                if query.table is None:
                    query.table = query.quality_table if self.run_config['quality_exps'] else  query.scalability_table
            
                for input_size_idx in range(len(self.run_config['input_sizes'])):
                    input_size = self.run_config['input_sizes'][input_size_idx] if query.class_name != "join" else self.run_config['input_sizes_for_join'][input_size_idx]
                    print(f"\n--- Running: {system_name.upper()} | {model_name} | {llm_provider} | Size: {input_size} ---")
                    
                    if self.run_config['wandb_report']:
                        wandb.init(
                            project = self.run_config['project_name'],
                            name=f"{system_name.lower()}_{query.task_name}_{query.id}_{llm_provider.lower()}_{model_name.lower()}_{input_size}",
                            group = query.class_name
                        )

                    try:
                        system_query = getattr(query, f"{system_name}_query", None)
                        if system_query is None:
                            raise ValueError(f"No query defined for {system_name} in query ID {query.id}")

                        query_kwargs = {}
                        if query.class_name == "derivation" and query.new_col_name is not None:
                            query_kwargs["new_col_name"] = query.new_col_name

                        if query.class_name == "join":
                            query_kwargs["table_left"] = input_folder + query.table_left
                            query_kwargs["cols_left"] = query.cols_left
                            query_kwargs["left_key"] = query.left_key
                            
                            query_kwargs["table_right"] = input_folder + query.table_right
                            query_kwargs["cols_right"] = query.cols_right
                            query_kwargs["right_key"] = query.right_key

                            query.table = ""    # Does not matter



                        output = system_instance.execute_query(
                            query.class_name,
                            system_query,
                            input_folder + query.table,
                            query.cols,
                            input_size,
                            **query_kwargs,
                        )
                        
                        predicted_table = output.get('result')
                        if self.run_config['quality_exps']:
                            quality = self._evaluate_results(query, predicted_table, input_size)
                        else:
                            quality = "-"

                        run_stats = {
                            "experiment": self.run_config['experiment_name'],
                            "system": system_name,
                            "llm_provider": llm_provider,
                            "model_name": model_name,
                            "input_size": input_size,
                            "query_id": query.id,
                            "class_name": query.class_name,
                            "task_name": query.task_name,
                            "latency_sec": output.get('latency'),
                            "input_tokens": output.get('input_tokens'),
                            "output_tokens": output.get('output_tokens'),
                            "total_tokens": output.get('total_tokens'),
                            "total_calls": output.get('total_calls'),
                            "tokens_throughput": output.get('tokens_throughput'),
                            "quality": quality,
                        }
                        
                        self.save_single_result(run_stats)

                        os.makedirs("../results", exist_ok=True)
                        os.makedirs("../results/outputs/", exist_ok=True)
                        if self.run_config['quality_exps']:
                            results_dir = "../results/outputs/quality"
                            os.makedirs(results_dir, exist_ok=True)
                        else:
                            results_dir = "../results/outputs/scalability"
                            os.makedirs("../results/outputs/scalability", exist_ok=True)

                        results_dir = f"{results_dir}/{llm_provider}_{model_name.replace('/', '_')}"
                        os.makedirs(results_dir, exist_ok=True)

                        predicted_table.to_csv(f"{results_dir}/{system_name}_{query.task_name}_{query.id}_{llm_provider}_{model_name.replace('/', '_')}_{input_size}.csv", index=False)

                        if self.run_config['wandb_report']:
                            wandb.log({
                                "predicted_table": wandb.Table(dataframe=predicted_table) if self.run_config['quality_exps'] else None,
                                "execution_time": output.get('latency'),
                                "input_tokens": output.get('input_tokens'),
                                "output_tokens": output.get('output_tokens'),
                                "total_tokens": output.get('total_tokens'),
                                "total_calls": output.get('total_calls'),
                                "tokens_throughput": output.get('tokens_throughput'),
                                "quality": quality
                            })

                            wandb.finish()

                    except Exception as e:
                        print(f"Error on {query.id}: {str(e)}")
                        pass

        print("\nAll experiments complete!")

    def save_single_result(self, result_dict: dict):
        """Saves a single run's statistics by appending to the CSV file immediately."""
        stats_dir = f"../results/stats/{'quality' if self.run_config['quality_exps'] else 'scalability'}"
        os.makedirs(stats_dir, exist_ok=True)
        filename = f"{stats_dir}/stats_{self.run_config['experiment_name']}.csv"

        file_exists = os.path.isfile(filename)

        df = pd.DataFrame([result_dict])
        df.to_csv(filename, mode='a', index=False, header=not file_exists)
        print(f"--> Saved stats for {result_dict['query_id']} to {filename}")

if __name__ == "__main__":
    runner = ExperimentRunner(
        run_config_path="configs/run_config.yaml",
        queries_path="configs/queries.yaml"
    )
    runner.run()