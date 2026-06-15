from typing import List
import os
import pandas as pd
from rapidfuzz import fuzz

# Helpers
def contains(row, col_a: str, col_b: str, threshold: int =85):
    attribute = row[col_a]
    text_field = row[col_b]

    if pd.isna(attribute) or pd.isna(text_field):
        return False
    
    attribute = str(attribute).lower()
    text_field = str(text_field).lower()

    correlation = fuzz.partial_ratio(attribute, text_field)

    return correlation >= threshold


def create_movie_dataset(drop_subset: List[str] =['movie_info', 'movie_title', 'rotten_tomatoes_link', 'runtime', 'original_release_date', 'directors', 'genres'], input_dir="datasets/rotten_tomatoes", output_dir="datasets/rotten_tomatoes/scalability_exps"):
    ### Derivation
    movies_original = pd.read_csv(f"{input_dir}/movies.csv").dropna(subset=drop_subset)
    movies = movies_original.copy()

    # Keep movies that their description mention at least two actor's real names
    movies['mentioned_actors'] = movies['movie_info'].str.findall(r'\((.*?)\)')
    movies['num_mentioned_actors'] = movies['mentioned_actors'].apply(len)
    movies = movies[movies['num_mentioned_actors'] > 1]

    final_df = movies[['rotten_tomatoes_link', 'movie_title', 'movie_info', 'genres', 'num_mentioned_actors', 'runtime', 'directors', 'original_release_date']].head(4000)
    final_df.to_csv(f"{output_dir}/movies_curated.csv", index=False)

    # Keep single-director movies and critic consensus contains director's name
    movies = movies[movies['directors'].str.split(", ").str.len() == 1]

    movies['director_in_description'] = movies.apply(lambda row: contains(row, 'directors', 'critics_consensus', threshold=85), axis=1)
    movies = movies[movies['director_in_description'] == True]

    # Duplicate the dataframe as the initial was 865 rows
    movies = pd.concat([movies] * 5, ignore_index=True)

    final_df = movies[['rotten_tomatoes_link', 'movie_title', 'movie_info', 'critics_consensus', 'genres', 'num_mentioned_actors', 'runtime', 'directors', 'original_release_date']].head(4000)
    final_df.to_csv(f"{output_dir}/movies_duplicated_for_directors.csv", index=False)

    # Specific query, multiple genres
    movies_multi_genres = movies_original[movies_original["genres"].str.split().str.len() > 1].head(4000)
    movies_multi_genres['secondary_genre'] = movies_multi_genres['genres'].str.split(',').str[1]
    movies_multi_genres[['rotten_tomatoes_link', 'movie_title', 'movie_info', 'genres', 'secondary_genre']].to_csv(f"{output_dir}/movies_multi_genres.csv")

    ### Selection
    movies_original[['rotten_tomatoes_link', 'movie_title', 'movie_info']].head(4000).to_csv(f"{output_dir}/actor_mentions.csv", index=False)
    
    

def create_reviews_dataset(drop_subset_reviews: List[str] = ['review_content'], drop_subset_movies: List[str] = ['movie_title'], input_dir="datasets/rotten_tomatoes", output_dir="datasets/rotten_tomatoes/scalability_exps"):
    reviews = pd.read_csv(f"{input_dir}/reviews.csv").dropna(subset=drop_subset_reviews)
    movies = pd.read_csv(f"{input_dir}/movies.csv").dropna(subset=drop_subset_movies)

    # Keep reviews that mention the title of the movie that refer to.
    movies_reviews = movies.merge(reviews, on='rotten_tomatoes_link')[['rotten_tomatoes_link', 'movie_title', 'review_content', 'review_type']]
    movies_reviews['title_in_review'] = movies_reviews.apply(lambda row: contains(row, 'movie_title', 'review_content', threshold=85), axis=1)
    movies_reviews = movies_reviews[movies_reviews['title_in_review'] == True].drop_duplicates(subset=['review_content'])

    # Is the review fresh?
    movies_reviews['is_fresh'] = movies_reviews['review_type'] == 'Fresh'
    
    movies_reviews.head(4000).to_csv(f"{output_dir}/reviews_curated.csv", index=False)

if __name__ == "__main__":
    os.makedirs("datasets/rotten_tomatoes/scalability_exps", exist_ok=True)

    create_movie_dataset()
    create_reviews_dataset()
    
    print("All pre-processing tasks completed successfully!")






