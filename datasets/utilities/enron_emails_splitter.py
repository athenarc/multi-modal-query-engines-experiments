import pandas as pd

sizes = [500, 1000, 2000, 4000]

for size in sizes:
    input_csv = pd.read_csv(f"datasets/enron_emails/enron_emails_shuffled_{size}.csv")

    for i, line in enumerate(input_csv["Message"]):
        if line:  # Only save non-empty lines
            with open(f"datasets/enron_emails/{size}/email{i}.txt", 'w', encoding='utf-8') as output_file:
                output_file.write(line)
