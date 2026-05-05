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

    def execute_query(self, class_name: str, query: str, table: str, cols: list, input_size: int) -> dict:
        if class_name == "derivation":
            input_df = pd.read_csv(f"../{table}")[cols].head(input_size)
            
            start_time = time.time()

            output_df = input_df.sem_map(query)
            output_df['winner'] = output_df['_map']

            execution_time = time.time() - start_time

            return {"result": output_df, "latency": execution_time}