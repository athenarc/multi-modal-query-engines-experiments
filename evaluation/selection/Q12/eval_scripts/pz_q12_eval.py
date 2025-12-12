import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

def count_true_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['Points_gt'] == df['Points_pred']) & (df['Points_gt'] == 15.0)])

def count_false_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['Points_gt'] != df['Points_pred']) & (df['Points_pred'] == 15.0)])

def count_true_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['Points_gt'] != 15.0)])

def count_false_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['Points_gt'] == 15.0)])

if __name__ == "__main__":
    player_labels = pd.read_csv("datasets/rotowire/player_labels.csv")
    reports_with_players = pd.read_csv("datasets/rotowire/reports_with_player_names/reports_with_players.csv").head(args.size)
    df_player_labels = player_labels.merge(reports_with_players, on=["Player Name", "Game ID"], how="inner")
    df_player_labels = player_labels[['Player Name', 'Points']]

    results_file = f"evaluation/selection/Q11/results/palimpzest_Q12_{args.model.replace(':', '_').replace('/', '_')}_{args.provider}_{args.size}.csv"
    pz_res = pd.read_csv(results_file)
    pz_res['Points'] = 15.0

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