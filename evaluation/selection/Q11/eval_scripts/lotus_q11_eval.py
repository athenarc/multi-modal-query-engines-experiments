import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
parser.add_argument("-o", "--opt", action='store_true', help="Evaluate optimized instead of default implementation of sem filet")
args = parser.parse_args()

def count_true_positives(df):
    if (df['_merge'] == 'both').any():
        return len(df[(df['_merge'] == 'both') & df.apply(lambda row: pd.notna(row['nationality_pred']) and pd.notna(row['nationality_gt']) and row['nationality_pred'] in row['nationality_gt'], axis=1)])
    return 0

def count_false_positives(df):
    if (df['_merge'] == 'both').any():
        return len(df[(df['_merge'] == 'both') & df.apply(lambda row: pd.notna(row['nationality_pred']) and pd.notna(row['nationality_gt']) and row['nationality_pred'] not in row['nationality_gt'] and row['nationality_pred'] == "American", axis=1)])
    return 0

def count_true_negatives(df):
    if (df['_merge'] == 'left_only').any():
        return len(df[(df['_merge'] == 'left_only') & (~df['nationality_gt'].str.contains("American", na=False))])
    return len(df[(df['nationality'] != 'American')])

def count_false_negatives(df):
    if (df['_merge'] == 'left_only').any():
        return len(df[(df['_merge'] == 'left_only') & (df['nationality_gt'].str.contains("American", na=False))])
    return len(df[df['nationality'] == 'American'])

if __name__ == "__main__":
    implementation = "cascades" if args.opt else "default"

    results_file = f"evaluation/selection/Q11/results/lotus_Q11_filter_{implementation}_{(args.model.replace('/', '_')).replace(':', '_')}_{args.provider}_{args.size}.csv"

    player_evidence = pd.read_csv("datasets/rotowire/player_evidence_mine.csv").head(args.size)
    player_evidence = player_evidence[['Player Name', 'nationality']]

    if os.stat(results_file).st_size == 0:
        lotus_res_default = pd.DataFrame(columns=['Player Name', 'nationality'])
    else:
        lotus_res_default = pd.read_csv(results_file)
        lotus_res_default = lotus_res_default.rename(columns={'player_name' : 'Player Name'})
        lotus_res_default['nationality'] = 'American'

    df = player_evidence.merge(lotus_res_default, on=['Player Name'], how='outer', suffixes=('_gt', '_pred'), indicator=True)

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