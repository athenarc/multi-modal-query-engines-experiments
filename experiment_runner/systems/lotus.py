from .base import BaseSystem
import time
import lotus
from lotus.models import LM
import pandas as pd

class LotusSystem(BaseSystem):
    def setup_llm(self):
        if self.llm_provider == 'ollama':
            self.lm = LM(self.llm_provider + '/' + self.model_name)
        elif self.llm_provider == 'vllm':
            self.lm = LM("hosted_vllm/" + self.model_name, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)
        lotus.settings.configure(lm=self.lm, enable_cache=False)        

        print(f"Lotus setup with {self.llm_provider} and model {self.model_name} completed.")

        if self.llm_provider == 'ollama':
            self._load_ollama_model()

    def _load_ollama_model(self):
        try:
            df_warmup = pd.DataFrame({"text": ["hello world"]})
            # Force an LLM call
            df_warmup.sem_map("Is this a greeting? {text}")
            print("Warm-up complete!")
        except Exception as e:
            print(f"Warm-up skipped/failed: {e}")

    def execute_derivation_query(
        self,
        nl_criterion: str,
        table: str,
        cols: list,
        input_size: int,
        new_col_name: str
    ) -> dict:
        input_df = pd.read_csv(f"../{table}")[cols].head(input_size)

        start_time = time.time()

        output_df = input_df.sem_map(nl_criterion)
        output_df[new_col_name] = output_df['_map']
        output_df.drop(columns=['_map'], inplace=True)

        execution_time = time.time() - start_time

        stats = lotus.settings.lm.stats
        
        input_tokens=stats.physical_usage.prompt_tokens
        output_tokens=stats.physical_usage.completion_tokens
        total_tokens=stats.physical_usage.total_tokens
        total_calls=input_size
        tokens_throughput = total_tokens / execution_time if execution_time > 0 else 0

        return {"result": output_df, 
                "latency": execution_time,
                "input_tokens": input_tokens, 
                "output_tokens": output_tokens, 
                "total_tokens": total_tokens, 
                "total_calls": total_calls, 
                "tokens_throughput": tokens_throughput
                }