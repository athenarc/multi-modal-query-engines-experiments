import pandas as pd

reports = pd.read_csv("datasets/rotowire/reports_table.csv").rename(columns={'Game_ID': 'Game ID'})
player_labels = pd.read_csv("datasets/rotowire/player_labels.csv")[['Game ID', 'Player Name']]
print(player_labels)

for size in [8, 14, 30]:
    reports_curr = reports[reports['Game ID'] < size]
    labels_curr = player_labels[player_labels['Game ID'] < size]

    reports_with_names = reports_curr.merge(labels_curr, on='Game ID', how='inner')
    reports_with_names.to_csv(f"datasets/rotowire/reports_with_player_names/reports_with_players_{size}.csv")