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

tp = len(df[df['_merge'] == 'both'])
fp = len(df[df['_merge'] == 'right_only'])
fn = len(df[df['_merge'] == 'left_only'])

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# with open('statistics/join/Q14.log', 'a') as file:
#     file.write(f"Accuracy: {accuracy:.2f}" + "\n")
#     file.write("------------------------------------------------------\n\n\n")