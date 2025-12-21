import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=10, const=10, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

df_movies_directors = pd.read_csv(f"datasets/movies_directors/movies_directors_split_{args.size}.csv")

results_file = f"evaluation/join/Q14/results/lotus_Q14_join_default_{args.model.replace(':', '_').replace('/', '_')}_{args.provider}_{args.size}.csv"

lotus_res = pd.read_csv(results_file, index_col=0)

df = df_movies_directors.merge(lotus_res, on=['title', 'director_name'], how='outer', indicator=True)

accuracy = len(df[df['_merge'] == 'both']) / lotus_res.shape[0]

print(f"Accuracy : {accuracy:.3f}")
