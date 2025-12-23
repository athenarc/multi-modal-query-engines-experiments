from datetime import datetime
import pandas as pd
import lotus
from lotus.models import LM
import wandb
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=50, const=50, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q14_join_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name=run_name,
        group="Join",
    )

df_papers = pd.read_csv("datasets/deepscholar-bench/paper_content.csv")[['arxiv_id', 'paper_title', 'related_works_section']].head(args.size)
df_cited_papers = pd.read_csv("datasets/deepscholar-bench/cited_papers_63.csv")

if args.provider == 'ollama':
    model = LM(args.provider + '/' + args.model)
elif args.provider == 'vllm':
    model = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)

lotus.settings.configure(lm=model)

instruction = "The paper {paper_title:left} cites the paper {cited_paper_title:right} in its related work section {related_works_section:left}."
start = time.time()
df = df_papers.sem_join(df_cited_papers, instruction)
df = df[['arxiv_id', 'cited_paper_title']]
exec_time = time.time() - start

output_file = f"evaluation/join/Q14/results/lotus_Q14_join_{args.model.replace(':', '_').replace('/', '_')}_{args.provider}_{args.size}.csv"
df.to_csv(output_file)
with open('statistics/join/Q14.log', 'a') as file:
    file.write(f"System: Lotus (sem_join -- default)\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write(f"Input Size: {args.size}\n")
    file.write(f"Execution Time: {exec_time:.2f}\n")
    file.write("Total LLM calls: " + str(args.size * 63) + "\n")


if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time,
        "total_LLM_calls": args.size*63
    })

    wandb.finish()