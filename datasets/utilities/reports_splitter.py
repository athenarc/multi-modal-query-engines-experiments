input_file = "datasets/rotowire/reports.txt"

with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

for i, line in enumerate(lines, start=1):
    if line:  # Only save non-empty lines
        with open(f"datasets/rotowire/reports_all/report_{i}.txt", 'w', encoding='utf-8') as output_file:
            output_file.write(line)
