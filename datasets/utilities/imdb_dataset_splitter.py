import pandas as pd

sizes = [100, 1000, 10000]

for size in sizes:
    input_csv = pd.read_csv("datasets/imdb_reviews/imdb_reviews.csv").head(size)

    for i, line in enumerate(input_csv["review"]):
        if line:  # Only save non-empty lines
            with open(f"datasets/imdb_reviews/{size}/review_{i}.txt", 'w', encoding='utf-8') as output_file:
                output_file.write(line)
