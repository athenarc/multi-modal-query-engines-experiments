from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
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

    def execute_query(
        self,
        class_name: str,
        nl_criterion: str,
        table: str,
        cols: list,
        input_size: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """Dispatch to the class-specific execute_<class_name>_query method."""
        method_name = f"execute_{class_name}_query"
        
        if not hasattr(self, method_name):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement {method_name}"
            )

        method = getattr(self, method_name)
        
        kwargs.setdefault("table", table)
        kwargs.setdefault("cols", cols)
        
        return method(nl_criterion, input_size, **kwargs)