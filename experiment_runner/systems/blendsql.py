from .base import BaseSystem
import time
import pandas as pd
from ollama import chat
from blendsql import BlendSQL
from blendsql.models import VLLM
from blendsql.ingredients import LLMMap, LLMQA, LLMJoin
from dotenv import load_dotenv

load_dotenv()

class BlendSQLSystem(BaseSystem):
    def setup_llm(self):
        # if self.llm_provider == 'ollama':
        #     self.model = LiteLLM(self.llm_provider + '/' + self.model_name, 
        #                     config={"timeout" : 50000, "cache": False},
        #                     caching=False,
        #                     env="/data/hdd1/users/jzerv/thesis_repo/.env")
        if self.llm_provider == 'vllm':
            self.model = VLLM(
                base_url="http://localhost:5001/v1",
                model=self.model_name,
                timeout=50000,
                cache=False
            )
        # elif self.llm_provider == 'transformers':
        #     self.model = TransformersLLM(
        #         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        #         config={"device_map": "auto"},
        #         caching=False,
        #         env="/data/hdd1/users/jzerv/thesis_repo/.env"
        #     )

        print(f"BlendSQL setup with {self.llm_provider} and model {self.model_name} completed.")
    
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
        new_col_type: str = 'str',
        **kwargs
    ) -> dict:
        input_df = pd.read_csv(f"../{table}")[cols].head(input_size)
        database = {
            "Input" : input_df
        }

        bsql = BlendSQL(
            db=database,
            model=self.model,
            verbose=True,
            ingredients={LLMMap},
        )

        selected_columns=", ".join(cols)

        output = bsql.execute(
            f"""
            SELECT {selected_columns}, {{{{
                LLMMAP(
                    '{nl_criterion}',
                    {selected_columns},
                    return_type='{new_col_type}',
                )
            }}}} AS {new_col_name}
            FROM Input
            """,
            infer_gen_constraints=False,
        )

        execution_time = output.meta.process_time_seconds
        input_tokens = output.meta.prompt_tokens
        output_tokens = output.meta.completion_tokens
        total_tokens = input_tokens + output_tokens
        total_calls = output.meta.num_generation_calls
        tokens_throughput = total_tokens / execution_time if execution_time > 0 else 0

        return {"result": output.df(),
                "latency": output.meta.process_time_seconds,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "total_calls": total_calls,
                "tokens_throughput": tokens_throughput
                }

    def execute_join_query(
        self,
        nl_criterion: str,
        input_size: list,
        table_left: str = None,
        cols_left: list = None,
        left_key: str = None,
        table_right: str = None,
        cols_right: list = None,
        right_key: str = None,
        **kwargs
    ) -> dict:
        input_df_left = pd.read_csv(f"../{table_left}")[cols_left].head(input_size[0])
        input_df_right = pd.read_csv(f"../{table_right}")[cols_right].head(input_size[1])
        
        database = {
            "LeftTable" : input_df_left,
            "RightTable" : input_df_right
        }

        bsql = BlendSQL(
            db=database,
            model=self.model,
            verbose=True,
            ingredients={LLMJoin},
        )

        output = bsql.execute(
            f"""
            SELECT *
            FROM LeftTable l
            JOIN RightTable r
            ON {{{{
                LLMJOIN(
                    l.{left_key},
                    r.{right_key},
                    join_criteria='{nl_criterion}',
                )
            }}}}
            """,
            infer_gen_constraints=False,
        )

        execution_time = output.meta.process_time_seconds
        input_tokens = output.meta.prompt_tokens
        output_tokens = output.meta.completion_tokens
        total_tokens = input_tokens + output_tokens
        total_calls = output.meta.num_generation_calls
        tokens_throughput = total_tokens / execution_time if execution_time > 0 else 0

        return {"result": output.df(),
                "latency": output.meta.process_time_seconds,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "total_calls": total_calls,
                "tokens_throughput": tokens_throughput
                }

    def execute_aggregation_query(
        self,
        nl_criterion: str,
        input_size: int,
        table: str = None,
        cols: list = None,
        **kwargs
    ) -> dict:
        input_df = pd.read_csv(f"../{table}")[cols].head(input_size)
        database = {
            "Input" : input_df
        }

        bsql = BlendSQL(
            db=database,
            model=self.model,
            verbose=True,
            ingredients={LLMQA},
        )

        selected_columns=", ".join(cols)

        output = bsql.execute(
            f"""
            SELECT DISTINCT {{{{
                LLMQA(
                    '{nl_criterion}',
                    {selected_columns},
                )
            }}}} AS result
            FROM Input
            """,
            infer_gen_constraints=False,
        )

        execution_time = output.meta.process_time_seconds
        input_tokens = output.meta.prompt_tokens
        output_tokens = output.meta.completion_tokens
        total_tokens = input_tokens + output_tokens
        total_calls = output.meta.num_generation_calls
        tokens_throughput = total_tokens / execution_time if execution_time > 0 else 0

        return {"result": output.df(),
                "latency": output.meta.process_time_seconds,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "total_calls": total_calls,
                "tokens_throughput": tokens_throughput
                }