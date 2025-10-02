input_file = "datasets/rotowire/reports/reports.txt"

with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

sizes = [8, 14, 30]

for size in sizes:
    for i, line in enumerate(lines[:size]):
        if line:  # Only save non-empty lines
            with open(f"datasets/rotowire/reports/{size}/report_{i}.txt", 'w', encoding='utf-8') as output_file:
                output_file.write(line)
