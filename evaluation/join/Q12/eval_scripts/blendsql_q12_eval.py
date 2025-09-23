import pandas as pd

df_movies = pd.read_csv("datasets/movies_dataset/movies.csv").head(10)
df_directors = pd.read_csv("datasets/movies_dataset/directors.csv")
df_movies_directors = df_movies.merge(df_directors, left_on=['director_id'], right_on=['id'])[['title', 'director_name']]

blendsql_res = pd.read_csv("join/Q12//results/blendsql_Q12_join_gemma3_12b_ollama.csv")[['title', 'director_name']]

df = df_movies_directors.merge(blendsql_res, on=['title', 'director_name'], how='outer', indicator=True)

accuracy = len(df[df['_merge'] == 'both']) / blendsql_res.shape[0]

print(f"Accuracy : {accuracy:.2f}")
