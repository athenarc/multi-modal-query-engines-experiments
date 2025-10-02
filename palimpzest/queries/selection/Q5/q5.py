import palimpzest as pz
from palimpzest.constants import Model
import pandas as pd
from dotenv import load_dotenv
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

model = f"{args.provider.upper()}_{args.model.replace(':', '_').replace('/', '_').replace('.', '_').replace('-', '_').upper()}"

load_dotenv()

if args.wandb:
    run_name=f"palimpzest_Q5_filter_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic selection",
    )

dataset = pz.TextFileDataset(id='imdb_reviews', path=f"datasets/imdb_reviews/{args.size}/")
dataset = dataset.sem_filter("The review is positive")

config = pz.QueryProcessorConfig(available_models=[Model.model])

output = dataset.run(config)

output_df = output.to_df()

if args.wandb:
    if args.provider == 'ollama':
        output_file = f"evaluation/selection/Q5/results/palimpzest_Q5_filter_{args.model.replace(':', '_')}_{args.provider}.csv"
    elif args.provider == 'vllm':
        output_file = f"evalution/selection/Q5/results/palimpzest_Q5_filter_{args.model.replace('/', '_')}_{args.provider}.csv"
    
    output_df.to_csv(output_file)
    
    wandb.log({
        "result_table": wandb.Table(dataframe=output_df),
        "execution_time": output.execution_stats.total_execution_time,
        "total_tokens": output.execution_stats.total_tokens
    })

    wandb.finish()
else:
    print("Result:\n\n", output_df)
    print("Execution time: ", output.executions_stats.total_execution_time)


