import pandas as pd
input_csv = pd.read_csv("datasets/rotowire/player_evidence_mine.csv").head(100)
player_names = input_csv["Player Name"]

print(input_csv.shape[0])

for i, line in enumerate(player_names):
    if line:  # Only save non-empty lines
        with open(f"datasets/rotowire/player_names/player{i}.txt", 'w', encoding='utf-8') as output_file:
            output_file.write(line)
