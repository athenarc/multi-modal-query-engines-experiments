import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
parser.add_argument("-o", "--opt", action='store_true', help="Evaluate optimized instead of default implementation of sem filet")
args = parser.parse_args()

def compute_metrics(df):
    tp = len(df[(df['_merge'] == 'both') & (df['Points_gt'] == df['Points_pred']) & (df['Points_gt'] == 15.0)])
    fp = len(df[(df['_merge'] == 'both') & (df['Points_gt'] != df['Points_pred']) & (df['Points_pred'] == 15.0)])
    tn = len(df[(df['_merge'] == 'left_only') & (df['Points_gt'] != 15.0)])
    fn = len(df[(df['_merge'] == 'left_only') & (df['Points_gt'] == 15.0)])
    return tp, fp, tn, fn


if __name__ == "__main__":
    implementation = "cascades" if args.opt else "default"

    player_labels = pd.read_csv("datasets/rotowire/player_labels.csv")
    reports_with_players = pd.read_csv("datasets/rotowire/reports_with_player_names/reports_with_players.csv")
    player_labels = player_labels.merge(reports_with_players, on=["Player Name", "Game ID"], how="inner")
    player_labels = player_labels[['Game ID', 'Player Name', 'Points']].head(args.size)

    results_file = f"evaluation/selection/Q12/results/lotus_Q12_filter_{implementation}_{(args.model.replace('/', '_')).replace(':', '_')}_{args.provider}_{args.size}.csv"

    lotus_res_default = pd.read_csv(results_file)
    lotus_res_default = lotus_res_default.drop(columns=["Report"])
    lotus_res_default['Points'] = 15.0

    df = player_labels.merge(lotus_res_default, on=["Game ID", "Player Name"], how="outer", suffixes=('_gt', '_pred'), indicator=True)

    tp, fp, tn, fn = compute_metrics(df)
    assert(tp + fp + tn + fn == args.size)


    with open('statistics/selection/Q12.log', 'a') as file:
        file.write(f"True Positives: {tp}\n")
        file.write(f"False Positives: {fp}\n")
        file.write(f"True Negatives: {tn}\n")
        file.write(f"False Negatives: {fn}\n")
        file.write(f"Accuracy: {(tp+tn) / (tp+tn+fp+fn):.2f}\n")
        file.write("------------------------------------------------------\n\n\n")