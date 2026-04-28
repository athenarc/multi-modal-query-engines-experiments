import pandas as pd
input_csv = pd.read_csv("datasets/rotowire/team_evidence.csv")
team_names = input_csv["Team Name"]

print(input_csv.shape[0])

for i, line in enumerate(team_names):
    if line:  # Only save non-empty lines
        with open(f"datasets/rotowire/team_names/team_{i}.txt", 'w', encoding='utf-8') as output_file:
            output_file.write(line)
