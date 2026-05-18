import yaml
import itertools
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional
from systems import get_system
from evaluation import evaluate_derivation_query

class Query(BaseModel):
    id: str
    class_name: str
    task_name: str
    nlq: str
    
    table: Optional[str] = None
    cols: Optional[List[str]] = None
    
    # Join queries
    table_left: Optional[str] = None
    cols_left: Optional[List[str]] = None
    table_right: Optional[str] = None
    cols_right: Optional[List[str]] = None
    left_key: str = None
    right_key: str = None
    
    new_col_name: Optional[str] = None
    new_col_type: Optional[str] = str
    lotus_query: Optional[str] = None
    palimpzest_query: Optional[str] = None
    blendsql_query: Optional[str] = None
    evaluation_cols: Optional[List[str]] = None

class ExperimentRunner:
    def __init__(self, run_config_path: str, queries_path: str):
        self.run_config = self._load_yaml(run_config_path)
        self.all_queries = [Query(**q) for q in self._load_yaml(queries_path)['queries']]
        self.results = []

    def _load_yaml(self, path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _evaluate_results(self, query: Query, predicted_df: pd.DataFrame) -> dict:
        """Evaluate predicted results against ground truth."""
        eval_metrics = {
            'accuracy': None,
            'partial_match_rate': None,
            'total_predictions': None,
            'total_correct': None,
            'total_partial': None,
            'quality': 'N/A',
            'eval_error': None
        }
        
        # Check if ground truth is configured for this query
        if not query.table or not query.evaluation_cols[-1]:
            eval_metrics['eval_error'] = "No ground truth configured for this query"
            return eval_metrics
        
        try:
            # Load ground truth on-the-fly
            ground_truth_df = pd.read_csv(query.table)
            
            # Evaluate based on query class
            if query.class_name == 'derivation':
                pred_col = query.new_col_name
                truth_col = query.ground_truth_col
                
                if not pred_col:
                    eval_metrics['eval_error'] = f"No predicted column specified for {query.class_name}"
                    return eval_metrics
                
                if pred_col not in predicted_df.columns:
                    eval_metrics['eval_error'] = f"Predicted column '{pred_col}' not found in results"
                    return eval_metrics
                
                if truth_col not in ground_truth_df.columns:
                    eval_metrics['eval_error'] = f"Ground truth column '{truth_col}' not found in {query.table}"
                    return eval_metrics
                
                results = evaluate_derivation_query(
                    query_id=query.id,
                    predicted_df=predicted_df,
                    ground_truth_df=ground_truth_df,
                    new_col_name=pred_col,
                    ground_truth_col_name=truth_col,
                    key_cols=None
                )
                
                eval_metrics['accuracy'] = results.get('accuracy')
                eval_metrics['partial_match_rate'] = results.get('partial_match_rate')
                eval_metrics['total_predictions'] = results.get('total_predictions')
                eval_metrics['total_correct'] = results.get('total_correct')
                eval_metrics['total_partial'] = results.get('total_partial')
                eval_metrics['eval_error'] = results.get('error')
                
                # Determine quality rating
                if eval_metrics['accuracy'] is not None:
                    if eval_metrics['accuracy'] >= 0.8:
                        eval_metrics['quality'] = 'excellent'
                    elif eval_metrics['accuracy'] >= 0.6:
                        eval_metrics['quality'] = 'good'
                    elif eval_metrics['accuracy'] >= 0.4:
                        eval_metrics['quality'] = 'fair'
                    else:
                        eval_metrics['quality'] = 'poor'
            else:
                eval_metrics['eval_error'] = f"Evaluation not yet implemented for {query.class_name}"
                
        except Exception as e:
            eval_metrics['eval_error'] = str(e)
        
        return eval_metrics

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

        for system_name, (llm_provider, model_name) in itertools.product(self.run_config['systems'], valid_llm_configs):
            # Initialize system once per LLM-Provider-Model combination
            system_instance = get_system(system_name, llm_provider, model_name)

            for query in queries_to_run:
                print(f"Executing Query {query.id} ({query.class_name} / {query.task_name})...")
            
                for input_size_idx in range(len(self.run_config['input_sizes'])):
                    input_size = self.run_config['input_sizes'][input_size_idx] if query.class_name != "join" else self.run_config['input_sizes_for_join'][input_size_idx]
                    print(f"\n--- Running: {system_name.upper()} | {model_name} | {llm_provider} | Size: {input_size} ---")
                    
                    try:
                        system_query = getattr(query, f"{system_name}_query", None)
                        if system_query is None:
                            raise ValueError(f"No query defined for {system_name} in query ID {query.id}")

                        query_kwargs = {}
                        if query.class_name == "derivation" and query.new_col_name is not None:
                            query_kwargs["new_col_name"] = query.new_col_name

                        if query.class_name == "join":
                            query_kwargs["table_left"] = query.table_left
                            query_kwargs["cols_left"] = query.cols_left
                            query_kwargs["left_key"] = query.left_key
                            
                            query_kwargs["table_right"] = query.table_right
                            query_kwargs["cols_right"] = query.cols_right
                            query_kwargs["right_key"] = query.right_key

                        output = system_instance.execute_query(
                            query.class_name,
                            system_query,
                            query.table,
                            query.cols,
                            input_size,
                            **query_kwargs,
                        )

                        # Evaluate results
                        predicted_result = output.get('result')
                        eval_metrics = self._evaluate_results(query, predicted_result)

                        # Log successful result
                        self.results.append({
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
                            "accuracy": eval_metrics.get('accuracy'),
                            "partial_match_rate": eval_metrics.get('partial_match_rate'),
                            "total_predictions": eval_metrics.get('total_predictions'),
                            "total_correct": eval_metrics.get('total_correct'),
                            "total_partial": eval_metrics.get('total_partial'),
                            "quality": eval_metrics.get('quality'),
                            "status": "success",
                            "error": None
                        })

                        predicted_result.to_csv(f"output_{query.id}_{system_name}_{llm_provider}_{model_name.replace('/', '_')}_{input_size}.csv", index=False)
                    except Exception as e:
                        print(f"Error on {query.id}: {str(e)}")
                        # Log failed result
                        self.results.append({
                            "experiment": self.run_config['experiment_name'],
                            "system": system_name,
                            "llm_provider": llm_provider,
                            "model_name": model_name,
                            "input_size": input_size,
                            "query_id": query.id,
                            "class_name": query.class_name,
                            "task_name": query.task_name,
                            "latency_sec": None,
                            "input_tokens": None,
                            "output_tokens": None,
                            "total_tokens": None,
                            "total_calls": None,
                            "tokens_throughput": None,
                            "accuracy": None,
                            "partial_match_rate": None,
                            "total_predictions": None,
                            "total_correct": None,
                            "total_partial": None,
                            "quality": "N/A",
                            "status": "failed",
                            "error": str(e)
                        })

            self.save_results()

    def save_results(self):
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        filename = f"results_{self.run_config['experiment_name']}.csv"
        df.to_csv(filename, index=False)
        print(f"\nExperiments complete! Results saved to {filename}")
        
        # Save detailed statistics
        self._save_statistics(df)
    
    def _save_statistics(self, results_df: pd.DataFrame):
        """Save aggregated statistics to a file."""
        stats_filename = f"statistics_{self.run_config['experiment_name']}.csv"
        
        # Group by system, class, task, query_id and aggregate metrics
        stat_cols = ['system', 'class_name', 'task_name', 'query_id']
        
        stats = results_df.groupby(stat_cols, as_index=False).agg({
            'latency_sec': ['mean', 'min', 'max', 'std'],
            'accuracy': ['mean', 'min', 'max'],
            'partial_match_rate': ['mean'],
            'total_predictions': ['first'],
            'total_correct': ['mean'],
            'total_partial': ['mean'],
            'quality': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A',
            'input_tokens': ['mean'],
            'output_tokens': ['mean'],
            'total_tokens': ['mean'],
            'tokens_throughput': ['mean'],
            'status': 'count'
        }).fillna('N/A')
        
        # Flatten column names
        stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in stats.columns.values]
        stats = stats.rename(columns={'status_count': 'run_count'})
        
        stats.to_csv(stats_filename, index=False)
        print(f"\nStatistics saved to {stats_filename}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(stats.to_string())


if __name__ == "__main__":
    runner = ExperimentRunner(
        run_config_path="configs/run_config.yaml",
        queries_path="configs/queries.yaml"
    )
    runner.run()