import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=1000, const=1000, type=int, help="The input size")
args = parser.parse_args()

imdb_reviews = pd.read_csv("datasets/imdb_reviews/imdb_reviews.csv").head(args.size)

imdb_reviews = imdb_reviews['sentiment'].value_counts()
positive_count = imdb_reviews.get('positive', 0)

with open('statistics/aggregation/Q16.log', 'a') as file:
    file.write(f"Ground Truth Answer: {positive_count}" + "\n")
    file.write("------------------------------------------------------\n\n\n")

