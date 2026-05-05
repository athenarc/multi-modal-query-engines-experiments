from abc import ABC, abstractmethod
from typing import Dict, Any
import time

class BaseSystem(ABC):
    def __init__(self, llm_provider: str, model_name: str):
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.setup_llm()

    @abstractmethod
    def setup_llm(self):
        """Initialize the connection to Ollama or vLLM."""
        pass

    @abstractmethod
    def execute_query(self, class_name: str, query: str, table: str, cols: list, input_size: int) -> Dict[str, Any]:
        """Execute the query and return results and execution time."""
        pass