import pandas as pd

movies = pd.read_csv("datasets/movies_directors/movies.csv", index_col=0)
directors = pd.read_csv("datasets/movies_directors/directors.csv", index_col=0)

merged = movies.merge(directors, left_on="director_id", right_on="id", how="inner", indicator=True)

# Get unique director names and then take the first 63 unique
directors_63 = merged['director_name'].drop_duplicates().head(63)

directors_63.to_csv("datasets/movies_directors/directors_63.csv", index=False)

for size in [10, 20, 30, 40, 50]:
    movies_ = merged.sample(size)[["title", "director_name"]]
    movies_.to_csv(f"datasets/movies_directors/movies_directors_split_{size}.csv", index=False)
