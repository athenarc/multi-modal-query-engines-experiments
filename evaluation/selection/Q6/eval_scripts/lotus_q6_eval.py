import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

def count_true_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['Points_gt'] == df['Points_pred']) & (df['Points_gt'] == 17.0)])

def count_false_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['Points_gt'] != df['Points_pred']) & (df['Points_pred'] == 17.0)])

def count_true_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['Points_gt'] != 17.0)])

def count_false_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['Points_gt'] == 17.0)])

if __name__ == "__main__":
    player_labels = pd.read_csv("datasets/rotowire/player_labels.csv")
    player_labels = player_labels[player_labels['Game ID'] < args.size]
    player_labels = player_labels[['Player Name', 'Points']]

    if args.provider == 'ollama':
        results_file = f"evaluation/selection/Q6/results/lotus_Q6_filter_default_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
    elif args.provider == 'vllm':
        results_file = f"evaluation/selection/Q6/results/lotus_Q6_filter_default_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

    lotus_res_default = pd.read_csv(results_file)
    lotus_res_default = lotus_res_default.drop(columns=["Report", "Game ID"])
    lotus_res_default['Points'] = 17.0

    df = player_labels.merge(lotus_res_default, on=["Player Name"], how="outer", suffixes=('_gt', '_pred'), indicator=True)

    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    print("--- Default Implementation ---")
    print(f"True Positives: {tp}"
        f"\nFalse Positives: {fp}"
        f"\nTrue Negatives: {tn}"
        f"\nFalse Negatives: {fn}")

    print(f"Accuracy for default implementation: {(tp + tn) / (tp + tn + fp + fn):.2f}")

    if args.provider == 'ollama':
        results_file = f"evaluation/selection/Q6/results/lotus_Q6_filter_cascades_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
    elif args.provider == 'vllm':
        exit(0)
        # results_file = f"evaluation/selection/Q5/results/lotus_Q9_filter_default_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

    lotus_res_opt = pd.read_csv(results_file)

    lotus_res_opt = lotus_res_opt.drop(columns=["Report", "Game ID"])
    lotus_res_opt['Points'] = 17.0

    df = player_labels.merge(lotus_res_opt, on=["Player Name"], how="outer", suffixes=('_gt', '_pred'), indicator=True)

    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    print("\n\n--- Optimized Implementation ---")
    print(f"True Positives: {tp}"
        f"\nFalse Positives: {fp}"
        f"\nTrue Negatives: {tn}"
        f"\nFalse Negatives: {fn}")
    print(f"Accuracy for optimized implementation: {(tp + tn) / (tp + tn + fp + fn):.2f}")