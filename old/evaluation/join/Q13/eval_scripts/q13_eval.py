import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=50, const=50, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
parser.add_argument("--system", nargs='?', default='lotus', const='lotus', type=str, help="The system that is being evaluated")
args = parser.parse_args()

banking_data = pd.read_csv("datasets/banking_data/banking_data_test.csv").head(args.size)


results_file = f"evaluation/join/Q13/results/{args.system}_Q13_join_{args.model.replace(':', '_').replace('/', '_')}_{args.provider}_{args.size}.csv"

result = pd.read_csv(results_file)

df = banking_data.merge(result, on=['text', 'category'], how='outer', indicator=True)
tp = len(df[df['_merge'] == 'both'])
fp = len(df[df['_merge'] == 'right_only'])
fn = len(df[df['_merge'] == 'left_only'])

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

with open('statistics/join/Q13.log', 'a') as file:
    file.write(f"Precision: {precision:.2f}" + "\n")
    file.write(f"Recall: {recall:.2f}" + "\n")
    file.write(f"F1 Score: {f1:.2f}" + "\n")
    file.write("------------------------------------------------------\n\n\n")