import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
parser.add_argument("-o", "--opt", action='store_true', help="Evaluate optimized instead of default implementation of sem filet")

args = parser.parse_args()

def count_true_positives(df):
    true_positives = df[(df['Spam/Ham_gt'] == 'spam') & (df['Spam/Ham_pred'] == 'spam')]
    return len(true_positives)

def count_false_positives(df):
    false_positives = df[(df['Spam/Ham_gt'] == 'ham') & (df['Spam/Ham_pred'] == 'spam')]
    return len(false_positives)

def count_true_negatives(df):
    true_negatives = df[(df['_merge'] == 'left_only') & (df['Spam/Ham_gt'] == 'ham')]
    return len(true_negatives)

def count_false_negatives(df):
    false_negatives = df[(df['_merge'] == 'left_only') & (df['Spam/Ham_gt'] == 'spam')]
    return len(false_negatives)


if __name__ == "__main__":
    implementation = "cascades" if args.opt else "default"

    results_file = f"evaluation/selection/Q10/results/lotus_Q10_filter_{implementation}_{(args.model.replace('/', '_')).replace(':', '_')}_{args.provider}_{args.size}.csv"

    enron_emails = pd.read_csv(f"datasets/enron_emails/enron_emails_shuffled_{args.size}.csv")[['Message ID', 'Message', 'Spam/Ham']]
    lotus_res = pd.read_csv(results_file, index_col=0)

    lotus_res["Spam/Ham"] = "spam"

    df = enron_emails.merge(lotus_res, on="Message", how="outer", suffixes=('_gt', '_pred'), indicator=True)

    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    with open('statistics/selection/Q10.log', 'a') as file:
        file.write(f"True Positives: {tp}\n")
        file.write(f"False Positives: {fp}\n")
        file.write(f"True Negatives: {tn}\n")
        file.write(f"False Negatives: {fn}\n")
        file.write(f"Accuracy: {(tp+tn) / (tp+tn+fp+fn):.2f}\n")
        file.write("------------------------------------------------------\n\n\n")