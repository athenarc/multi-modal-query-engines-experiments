import pandas as pd

sizes = [100, 250, 500]

for size in sizes:
    input_csv = pd.read_csv("datasets/rotowire/player_evidence_mine.csv").head(size)
    player_names = input_csv["Player Name"]

    for i, line in enumerate(player_names):
        if line:  # Only save non-empty lines
            with open(f"datasets/rotowire/player_names/{size}/player{i}.txt", 'w', encoding='utf-8') as output_file:
                output_file.write(line)
