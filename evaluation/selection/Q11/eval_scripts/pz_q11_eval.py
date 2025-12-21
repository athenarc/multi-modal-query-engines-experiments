import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

def compute_metrics(df):
    negative_gt = df['label_gt'].isin(['NOT ENOUGH INFO', 'REFUTES'])

    tp = len(df[(df['label_gt'] == 'SUPPORTS') & (df['label_pred'] == 'SUPPORTS')])
    fp = len(df[negative_gt & (df['label_pred'] == 'SUPPORTS')])
    tn = len(df[negative_gt &(df['label_pred'] != 'SUPPORTS')])
    fn = len(df[(df['label_gt'] == 'SUPPORTS') &(df['label_pred'] != 'SUPPORTS')])

    return tp, fp, tn, fn

if __name__ == "__main__":
    results_file = f"evaluation/selection/Q11/results/palimpzest_Q11_{args.model.replace(':', '_').replace('/', '_')}_{args.provider}_{args.size}.csv"

    claims = pd.read_csv("datasets/fever/fever.csv")[['id', 'claim', 'label']].head(args.size)

    pz_res = pd.read_csv(results_file, index_col=0)

    if (pz_res.empty):
        pz_res = pd.DataFrame(columns=['id', 'claim', 'label'])
    else:
        pz_res['label'] = "SUPPORTS"

    df = claims.merge(pz_res, on='id', how='outer', suffixes=('_gt', '_pred'), indicator=True)

    tp, fp, tn, fn = compute_metrics(df)
    assert(tp+tn+fp+fn == args.size)

    with open('statistics/selection/Q11.log', 'a') as file:
        file.write(f"True Positives: {tp}\n")
        file.write(f"False Positives: {fp}\n")
        file.write(f"True Negatives: {tn}\n")
        file.write(f"False Negatives: {fn}\n")
        file.write(f"Accuracy: {(tp+tn) / (tp+tn+fp+fn):.2f}\n")
        file.write("------------------------------------------------------\n\n\n")