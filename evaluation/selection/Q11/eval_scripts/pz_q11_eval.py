import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

def count_true_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['nationality_gt'] == df['nationality_pred']) & (df['nationality_gt'] == "American")])

def count_false_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['nationality_gt'] != df['nationality_pred']) & (df['nationality_pred'] == "American")])

def count_true_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['nationality_gt'] != "American")])

def count_false_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['nationality_gt'] == "American")])

if __name__ == "__main__":
    if args.provider == 'ollama':
        results_file = f"evaluation/selection/Q11/results/palimpzest_Q11_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
    elif args.provider == 'vllm':
        results_file = f"evaluation/selection/Q11/results/palimpzest_Q11_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

    df_player_labels = pd.read_csv("datasets/rotowire/player_evidence_mine.csv").head(args.size)
    df_player_labels = df_player_labels[['Player Name', 'nationality']]

    pz_res = pd.read_csv(results_file)
    pz_res['nationality'] = 'American'

    df = df_player_labels.merge(pz_res, on='Player Name', how='outer', suffixes=('_gt', '_pred'), indicator=True)

    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    with open('statistics/selection/Q11.log', 'a') as file:
        file.write(f"True Positives: {tp}\n")
        file.write(f"False Positives: {fp}\n")
        file.write(f"True Negatives: {tn}\n")
        file.write(f"False Negatives: {fn}\n")
        file.write(f"Accuracy: {(tp+tn) / (tp+tn+fp+fn):.2f}\n")
        file.write("------------------------------------------------------\n\n\n")