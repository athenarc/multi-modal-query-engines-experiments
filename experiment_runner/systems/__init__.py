from .lotus import LotusSystem
# from .palimpzest import PalimpzestSystem
# from .blendsql import BlendSQLSystem

def get_system(system_name: str, llm_provider: str, model_name: str):
    systems = {
        "lotus": LotusSystem,
        # "palimpzest": PalimpzestSystem,
        # "blendsql": BlendSQLSystem
    }
    if system_name not in systems:  
        raise ValueError(f"System {system_name} not supported.")
    return systems[system_name](llm_provider, model_name)