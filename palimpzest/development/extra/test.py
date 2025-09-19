import palimpzest as pz
from palimpzest.constants import Model
import pandas as pd
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()

    df = pd.DataFrame(
        {
            "description": [
                "Ronaldinho is 30 years old and his height is 180 cm",
                "Messi is 35 years old and his height is 1.7 m",
                "C.Ronaldo is 38 years old and his height is 185 cm",
            ]
        }
    )

    input_reviews = df["description"]

    dataset = pz.Dataset(pd.DataFrame(input_reviews))
    print(dataset.project(project_cols=["description"]))
    dataset = dataset.sem_add_columns(
        [
            {
                "name": "age",
                "type": int,
                "desc": "The age of the player in years.",
            },
            {
                "name": "height",
                "type": int,
                "desc": "The height of the player in centimeters.",
            },

        ]
    )

    config = pz.QueryProcessorConfig(policy=pz.MaxQuality())
    # config = pz.QueryProcessorConfig(policy=pz.MinCost())
    # config = pz.QueryProcessorConfig(policy=pz.MinTime())
    # config = pz.QueryProcessorConfig(policy=pz.MaxQualityAtFixedCost)

    output = dataset.run(config)

    print("Cost:", output.execution_stats.total_execution_cost, "\n\n")
    print("Time:", output.execution_stats.total_execution_time, "\n\n")
    output_df = output.to_df(cols=["description", "age", "height"])
    print(output_df)