import palimpzest as pz
from palimpzest.constants import Model
import pandas as pd
from dotenv import load_dotenv
import wandb
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

model = getattr(Model, f"{args.provider.upper()}_{args.model.replace(':', '_').replace('/', '_').replace('.', '_').replace('-', '_').upper()}")

load_dotenv()

if args.wandb:
    run_name=f"palimpzest_Q10_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name=run_name,
        group="Selection",
    )

dataset = pz.TextFileDataset(id='enron_emails', path=f"datasets/enron_emails/{args.size}/")

dataset = dataset.sem_filter("The email is spam")

config = pz.QueryProcessorConfig(available_models=[model])
output = dataset.run(config)

output_df = output.to_df()

output_file = f"evaluation/selection/Q10/results/palimpzest_Q10_{args.model.replace(':', '_').replace('/', '_')}_{args.provider}_{args.size}.csv"
output_df.to_csv(output_file)

with open('statistics/selection/Q10.log', 'a') as file:
    file.write(f"System: Palimpzest\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Execution Time: {output.execution_stats.total_execution_time:.2f}\n")
    # file.write(f"Total tokens: {output.execution_stats.total_tokens}")

if args.wandb:    
    wandb.log({
        "result_table": wandb.Table(dataframe=output_df),
        "execution_time": output.execution_stats.total_execution_time,
        # "total_tokens": output.execution_stats.total_tokens
    })
    wandb.finish()