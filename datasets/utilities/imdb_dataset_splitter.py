import pandas as pd
input_csv = pd.read_csv("datasets/imdb_reviews/imdb_reviews.csv").head(1000)

print(input_csv.shape[0])

for i, line in enumerate(input_csv["review"]):
    if line:  # Only save non-empty lines
        with open(f"datasets/imdb_reviews/imdb_reviews_1000/review_{i}.txt", 'w', encoding='utf-8') as output_file:
            output_file.write(line)
