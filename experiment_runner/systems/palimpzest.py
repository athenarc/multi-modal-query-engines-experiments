from .base import BaseSystem
import palimpzest as pz
from palimpzest.constants import Model, PromptStrategy

import pandas as pd
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

class PalimpzestSystem(BaseSystem):
    def setup_llm(self):
        self.model = getattr(Model, f"{self.llm_provider.upper()}_{self.model_name.replace(':', '_').replace('/', '_').replace('.', '_').replace('-', '_').upper()}")

        if (self.llm_provider == 'ollama'):
            self._load_ollama_model()

        print(f"Palimpzest setup with {self.llm_provider} and model {self.model_name} completed.")

    def _load_ollama_model(self):
        try:
            chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Hello'
                    }
                ]
            )
            print("Warm-up complete!")
        except Exception as e:
            print(f"Warm-up skipped/failed: {e}")

    def _compute_stats(self, output):
        execution_time = output.execution_stats.total_execution_time
        input_tokens=0
        output_tokens=0
        total_calls=0

        # Palimpzest's execution stats are nested, so we need to iterate through them to get the token counts and LLM call counts
        for _, plan_stats in output.execution_stats.plan_stats.items():
            for _, op_stats in plan_stats.operator_stats.items():
                input_tokens += op_stats.input_text_tokens
                output_tokens += op_stats.output_text_tokens
        
                for record_stat in op_stats.record_op_stats_lst:
                    total_calls += getattr(record_stat, 'total_llm_calls', 0)

        total_tokens = input_tokens + output_tokens
        tokens_throughput = total_tokens / execution_time if execution_time > 0 else 0

        return {"result": output.to_df(), 
                "latency": execution_time,
                "input_tokens": input_tokens, 
                "output_tokens": output_tokens, 
                "total_tokens": total_tokens, 
                "total_calls": total_calls,
                "tokens_throughput": tokens_throughput
                }


    def execute_derivation_query(
        self,
        nl_criterion: str,
        input_size: int,
        table: str = None,
        cols: list = None,
        new_col_name: str = None,
        new_col_type: type = str,
        **kwargs
    ) -> dict:
        input_df = pd.read_csv(table)[cols].head(input_size)
        summaries = pz.MemoryDataset(
            id=f"{table.split('/')[-1].split('.')[0]}_{hash(str(input_df.index.tolist()))}",
            vals=input_df
        )

        output = summaries.sem_add_columns(
            cols=[
                {
                    "name": new_col_name,
                    "type": new_col_type,
                    "desc": nl_criterion,
                },
            ],
            depends_on=cols,
        )

        config = pz.QueryProcessorConfig(
            available_models=[self.model],
            timeout=50000,
            execution_strategy="parallel",
            max_workers=1,
            reasoning_effort="disable"
        )
        output = output.run(config=config)

        return self._compute_stats(output)

    def execute_selection_query(
        self,
        nl_criterion: str,
        input_size: int,
        table: str = None,
        cols: list = None,
        **kwargs
    ) -> dict:
        input_df = pd.read_csv(table)[cols].head(input_size)
        input_table = pz.MemoryDataset(id=table.split('/')[-1].split('.')[0], vals=input_df)

        output = input_table.sem_filter(
            nl_criterion,
            depends_on=cols,
        )

        config = pz.QueryProcessorConfig(
            available_models=[self.model],
            timeout=50000,
            execution_strategy="parallel",
            max_workers=16,
            reasoning_effort="disable"
        )
        output = output.run(config=config)
        
        return self._compute_stats(output)
    
    def execute_join_query(
            self,
            nl_criterion: str,
            input_size,
            table_left: str = None,
            cols_left: list = None,
            table_right: str = None,
            cols_right: list = None,
            left_key: str = None,
            right_key: str = None,
            **kwargs
    ) -> dict:
        if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
            size_left, size_right = input_size[0], input_size[1]
        else:
            size_left = size_right = input_size

        input_df_left = pd.read_csv(table_left)[cols_left].head(size_left)
        input_df_right = pd.read_csv(table_right)[cols_right].head(size_right)

        input_table_left = pz.MemoryDataset(id=table_left.split('/')[-1].split('.')[0], vals=input_df_left)
        input_table_right = pz.MemoryDataset(id=table_right.split('/')[-1].split('.')[0], vals=input_df_right)

        output = input_table_left.sem_join(input_table_right, condition=nl_criterion)
        
        config = pz.QueryProcessorConfig(
            available_models=[self.model],
            timeout=50000,
            execution_strategy="parallel",
            max_workers=16,
            join_parallelism=16,
            reasoning_effort="disable"
        )
        output = output.run(config=config)

        return self._compute_stats(output)
    
    def execute_aggregation_query(
        self,
        nl_criterion: str,
        input_size: int,
        table: str = None,
        cols: list = None,
        new_col_name: str = None,
        new_col_type: type = str,
        **kwargs
    ) -> dict:
        input_df = pd.read_csv(table)[cols].head(input_size)
        input_table = pz.MemoryDataset(id=table.split('/')[-1].split('.')[0], vals=input_df)

        print("new_colname:", new_col_name)
        print("new_col_type:", new_col_type)

        output = input_table.sem_agg(
            col={
                "name": new_col_name, 
                "desc": nl_criterion, 
                "type": new_col_type
            },
            agg=nl_criterion,
        )

        config = pz.QueryProcessorConfig(
            available_models=[self.model],
            timeout=50000,
            execution_strategy="parallel",
            max_workers=16,
            reasoning_effort="disable",
        )
        output = output.run(config=config)


        return self._compute_stats(output)