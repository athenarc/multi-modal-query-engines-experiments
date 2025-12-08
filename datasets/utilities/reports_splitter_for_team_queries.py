import os

input_file = "datasets/rotowire/reports/reports.txt"

# The indices of the rows you want to exclude
exclude_indices = [8, 39, 68, 82, 122, 123, 150, 155, 192, 199, 211, 214, 255, 267, 274, 290, 294, 313, 330, 343, 345, 363, 379, 391, 398, 423, 439, 472, 499, 500, 534, 558, 562, 565, 568, 570, 644, 645, 668, 681, 721]

sizes = [10, 50, 100, 200]

with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

valid_entries = []
for index, line in enumerate(lines):
    if index not in exclude_indices:    # Exclude games that we do not have ground truth values
        valid_entries.append((index, line))

for size in sizes:
    current_batch = valid_entries[:size]
    
    for original_index, line in current_batch:
        if line: 
            output_path = f"datasets/rotowire/reports/for_team_queries/{size}/report_{original_index}.txt"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(line)