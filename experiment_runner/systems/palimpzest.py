from .base import BaseSystem

import palimpzest as pz
from palimpzest.constants import Model
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
        input_df = pd.read_csv(f"../{table}")[cols].head(input_size)
        summaries = pz.MemoryDataset(id=table.split('/')[-1].split('.')[0], vals=input_df)

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
        )
        output = output.run(config=config)

        print(output.execution_stats)
        
        execution_time = output.execution_stats.total_execution_time
        input_tokens=output.execution_stats.total_input_tokens
        output_tokens=output.execution_stats.total_output_tokens
        total_tokens=output.execution_stats.total_tokens
        total_calls=input_size
        tokens_throughput = total_tokens / execution_time if execution_time > 0 else 0

        return {"result": output.to_df(), 
                "latency": execution_time,
                "input_tokens": input_tokens, 
                "output_tokens": output_tokens, 
                "total_tokens": total_tokens, 
                "total_calls": total_calls,
                "tokens_throughput": tokens_throughput
                }