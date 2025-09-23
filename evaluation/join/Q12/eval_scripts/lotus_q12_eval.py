import pandas as pd

df_movies = pd.read_csv("datasets/movies_dataset/movies.csv").head(10)
df_directors = pd.read_csv("datasets/movies_dataset/directors.csv")
df_movies_directors = df_movies.merge(df_directors, left_on=['director_id'], right_on=['id'])[['title', 'director_name']]

lotus_res = pd.read_csv("join/Q12/results/lotus_Q12_join_default_gemma3_12b_ollama.csv")[['title', 'director_name']]

df = df_movies_directors.merge(lotus_res, on=['title', 'director_name'], how='outer', indicator=True)

accuracy = len(df[df['_merge'] == 'both']) / lotus_res.shape[0]

print(f"Accuracy : {accuracy:.3f}")
