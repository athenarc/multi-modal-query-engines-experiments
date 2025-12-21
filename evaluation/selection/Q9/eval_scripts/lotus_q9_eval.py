import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
parser.add_argument("-o", "--opt", action='store_true', help="Evaluate optimized instead of default implementation of sem filet")

args = parser.parse_args()

def compute_metrics(df):
    tp = len(df[(df['sentiment_gt'] == 'positive') & (df['sentiment_pred'] == 'positive')])
    fp = len(df[(df['sentiment_gt'] == 'negative') & (df['sentiment_pred'] == 'positive')])
    tn = len(df[(df['_merge'] == 'left_only') & (df['sentiment_gt'] == 'negative')])
    fn = len(df[(df['_merge'] == 'left_only') & (df['sentiment_gt'] == 'positive')])
    return tp, fp, tn, fn

if __name__ == "__main__":
    implementation = "cascades" if args.opt else "default"

    results_file = f"evaluation/selection/Q9/results/lotus_Q9_filter_{implementation}_{(args.model.replace('/', '_')).replace(':', '_')}_{args.provider}_{args.size}.csv"

    imdb_dataset = pd.read_csv("datasets/imdb_reviews/imdb_reviews.csv").drop_duplicates().head(args.size)

    lotus_res = pd.read_csv(results_file, index_col=0)

    lotus_res["sentiment"] = "positive"

    df = imdb_dataset.merge(lotus_res, on="review", how="outer", suffixes=('_gt', '_pred'), indicator=True)

    tp, fp, tn, fn = compute_metrics(df)
    assert(tp+tn+fp+fn == args.size)

    with open('statistics/selection/Q9.log', 'a') as file:
        file.write(f"True Positives: {tp}\n")
        file.write(f"False Positives: {fp}\n")
        file.write(f"True Negatives: {tn}\n")
        file.write(f"False Negatives: {fn}\n")
        file.write(f"Accuracy: {(tp+tn) / (tp+tn+fp+fn):.2f}\n")
        file.write("------------------------------------------------------\n\n\n")