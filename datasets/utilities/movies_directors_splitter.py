import pandas as pd

movies = pd.read_csv("datasets/movies_directors/movies.csv", index_col=0)
directors = pd.read_csv("datasets/movies_directors/directors.csv", index_col=0)

merged = movies.merge(directors, left_on="director_id", right_on="id", how="inner", indicator=True)

directors_63 = merged['director_name'].head(63)

directors_63.to_csv("datasets/movies_directors/directors_63.csv", index=False)

for size in [10, 25, 40, 50, 60]:
    movies_ = merged.sample(size)[["title", "director_name"]]
    movies_.to_csv(f"datasets/movies_directors/movies_directors_split_{size}.csv", index=False)