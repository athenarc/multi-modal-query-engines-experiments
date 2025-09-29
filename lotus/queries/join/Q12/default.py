import pandas as pd
import lotus
from lotus.models import LM
import wandb
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=10, const=10, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q12_join_default_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic join",
    )

df_movies = pd.read_csv("datasets/movies_directors/movies.csv").head(args.size)[['title']]
df_directors = pd.read_csv("datasets/movies_directors/directors.csv")[['director_name']]

if args.provider == 'ollama':
    model = args.provider + '/' + args.model

lm = LM(model=model)
lotus.settings.configure(lm=lm)

instruction = "The movie {title:left} is directed by {director_name:right}."
start = time.time()
df = df_movies.sem_join(df_directors, instruction)
exec_time = time.time() - start

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", df)
    print("Execution time: ", exec_time)