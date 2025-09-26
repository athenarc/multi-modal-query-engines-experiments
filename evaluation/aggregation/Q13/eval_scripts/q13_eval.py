import pandas as pd

imdb_reviews = pd.read_csv("datasets/imdb_reviews/imdb_reviews.csv").head(10000)

imdb_reviews = imdb_reviews['sentiment'].value_counts()

print(imdb_reviews)