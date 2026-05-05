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
        lotus.settings.configure(lm=self.lm)        

        print(f"Lotus setup with {self.llm_provider} and model {self.model_name} completed.")

    def execute_query(self, class_name: str, query: str, table: str, cols: list, input_size: int) -> dict:
        if class_name == "derivation":
            input_df = pd.read_csv(f"../{table}")[cols].head(input_size)
            
            start_time = time.time()

            output_df = input_df.sem_map(query)
            output_df['winner'] = output_df['_map']

            execution_time = time.time() - start_time

            return {"result": output_df, "latency": execution_time}

            

        # start_time = time.time()
        
        # # TODO: Insert actual Lotus API calls here
        # # Example: df.lotus.filter(nlq)
        # result_data = f"Mock Lotus result for {input_size} rows"
        
        # execution_time = time.time() - start_time
        # return {"result": result_data, "latency": execution_time} 