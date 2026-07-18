from typing import List
import os
import pandas as pd
from rapidfuzz import fuzz

# Helpers
def contains(row, col_a: str, col_b: str, threshold: int = 85):
    attribute = row[col_a]
    text_field = row[col_b]

    if pd.isna(attribute) or pd.isna(text_field):
        return False
    
    attribute = str(attribute).lower()
    text_field = str(text_field).lower()

    correlation_score = fuzz.partial_ratio(attribute, text_field)

    return correlation_score >= threshold


def create_movie_dataset(drop_subset: List[str] =['movie_info', 'movie_title', 'rotten_tomatoes_link', 'runtime', 'original_release_date', 'directors', 'genres'], input_dir="datasets/rotten_tomatoes", output_dir="datasets/rotten_tomatoes/quality_exps"):
    ### Derivation curation

    movies = pd.read_csv(f"{input_dir}/movies.csv").dropna(subset=drop_subset)

    # Specific query, multiple genres
    movies_multi_genres = movies[movies["genres"].str.split().str.len() > 1].head(100)
    movies_multi_genres['secondary_genre'] = movies_multi_genres['genres'].str.split(',').str[1]
    movies_multi_genres[['rotten_tomatoes_link', 'movie_title', 'movie_info', 'genres', 'secondary_genre']].to_csv(f"{output_dir}/movies_multi_genres.csv")

    # Keep single-genre movies
    movies = movies[movies["genres"].str.split().str.len() == 1]
    movies["genre"] = movies["genres"].apply(lambda x: x if x in ["Drama", "Comedy", "Action", "Horror", "Romance", "Thriller", "Documentary", "Animation"] else "Other")

    # Column of mentioned actors
    movies['mentioned_actors'] = movies['movie_info'].str.findall(r'\((.*?)\)').apply(lambda groups:[name.strip() for g in groups for name in g.split(",")])
    movies['num_mentioned_actors'] = movies['mentioned_actors'].apply(len)

    # Lead actors
    movies = movies[movies['num_mentioned_actors'] > 0]
    movies['lead_actor'] = movies['actors'].str.split(", ").str[0]

    # Second real fullname mentioned
    movies = movies[movies['num_mentioned_actors'] > 1]
    movies['second_fullname'] = movies['mentioned_actors'].str[1]

    # Keep single-director movies and critic consensus contains director's name
    movies = movies[movies['directors'].str.split(", ").str.len() == 1]
    
    movies['director_in_description'] = movies.apply(lambda row: contains(row, 'directors', 'critics_consensus', threshold=85), axis=1)
    movies = movies[movies['director_in_description'] == True]

    movies = movies[movies['content_rating'].isin(['G', 'PG', 'R'])]

    movies.loc[movies['movie_title'] == "Bobby", 'lead_actor'] = "Ben Affleck"
    movies.loc[movies['movie_title'] == "A Bronx Tale", 'lead_actor'] = "Lillo Brancato"
    movies.loc[movies['movie_title'] == "Rain Man", 'lead_actor'] = "Tom Cruise"
    movies.loc[movies['movie_title'] == "The Company Men", 'lead_actor'] = "Ben Affleck"
    movies.loc[movies['movie_title'] == "Heavy", 'lead_actor'] = "Pruitt Tayolor Vince"
    movies.loc[movies['movie_title'] == "Red Hook Summer", 'lead_actor'] = "Jules Brown"
    movies.loc[movies['movie_title'] == "Stake Land", 'lead_actor'] = "Nnick Damici"

    excluded_titles = [
        'Intolerance',
        'Heaven & Earth',
        'Chi-Raq',
        'Closer',
        'Get on the Bus',
        'Halloween',
        'Hard Eight',
        'Animal House',
        'Used Cars',
        "National Lampoon's Animal House",
        'Ordinary People',
        'Silence',
        'The Bling Ring',
        'Explicit Ills',
        'History of the World---Part I',
        'Results',
        'The Fog',
        'The Hundred-Foot Journey',
        'The Program',
        'Five Minutes of Heaven',
        'Fanny & Alexander',
        'Higher Learning',
        'Lost in America',
        'The Merry Gentleman',
        'Mother and Child',
        'Night Moves',
        'Some Velvet Morning',
        'The Family Fang',
        'The Hateful Eight',
        'Thelma & Louise',
        'This Is 40',
        'To Rome with Love',
        "The Blackcoat's Daughter",
        'The Place Beyond The Pines',
        'Wag the Dog',
        'Wall Street',
        'Wanderlust',
        'Wiener-Dog',
        'Women in Trouble',
    ]
    movies = movies[~movies['movie_title'].isin(excluded_titles)]

    movies['release_year'] = pd.to_datetime(movies['original_release_date'], errors="coerce").dt.year

    final_df = movies[['rotten_tomatoes_link', 'movie_title', 'movie_info', 'critics_consensus', 'genre', 'num_mentioned_actors', 'lead_actor', 'second_fullname', 'runtime', 'directors', 'original_release_date', 'release_year', 'content_rating']].head(100)
    final_df.to_csv(f"{output_dir}/movies_derivation_curated.csv", index=False)

    ### Selection Curation
    movies_classification_extknowledge = final_df[['rotten_tomatoes_link', 'movie_title', 'movie_info', 'genre', 'runtime', 'original_release_date']].copy()

    movies_classification_extknowledge['is_drama'] = movies_classification_extknowledge['genre'].str.lower() == "drama"
    movies_classification_extknowledge['is_comedy'] = movies_classification_extknowledge['genre'].str.lower() == "comedy"
    movies_classification_extknowledge["before_2000"] = pd.to_datetime(movies_classification_extknowledge["original_release_date"], errors="coerce").dt.year < 2000
    movies_classification_extknowledge["over_2h"] = movies_classification_extknowledge['runtime'] > 120.0

    movies_variation = pd.read_csv("rotten_tomatoes_movies_variation.csv")[['title', 'originalLanguage']].drop_duplicates(subset='title')
    movies_classification_extknowledge_expanded = movies_classification_extknowledge.merge(movies_variation, left_on='movie_title', right_on='title', how="left").drop(columns=['title'])

    movies_classification_extknowledge_expanded.loc[movies_classification_extknowledge_expanded["movie_title"] == "Death in Venice", "originalLanguage"] = "German"
    movies_classification_extknowledge_expanded.loc[movies_classification_extknowledge_expanded["movie_title"] == "Intolerance", "originalLanguage"] = "English"
    movies_classification_extknowledge_expanded.loc[movies_classification_extknowledge_expanded["movie_title"] == "Fanny & Alexander", "originalLanguage"] = "Swedish"
    movies_classification_extknowledge_expanded.loc[movies_classification_extknowledge_expanded["movie_title"] == "History of the World---Part I", "originalLanguage"] = "English"
    movies_classification_extknowledge_expanded.loc[movies_classification_extknowledge_expanded["movie_title"] == "Letters from Iwo Jima", "originalLanguage"] = "Japanese"
    movies_classification_extknowledge_expanded.loc[movies_classification_extknowledge_expanded["movie_title"] == "Mother Of George", "originalLanguage"] = "English"

    movies_classification_extknowledge_expanded["written_in_german"] = movies_classification_extknowledge_expanded['originalLanguage'].str.lower() == "german"

    movies_classification_extknowledge_expanded.to_csv(f"{output_dir}/movies_class_ext_selection_curated.csv", index=False)

    movies_reasoning = pd.read_csv(f"{input_dir}/movies.csv").dropna(subset=drop_subset)[['rotten_tomatoes_link', 'movie_title', 'movie_info', 'actors']]
    
    # The movie description mentions some of the actor's real names
    movies_reasoning['mentioned_actors'] = movies_reasoning['movie_info'].str.findall(r'\((.*?)\)').apply(lambda groups:[name.strip() for g in groups for name in g.split(",")])
    movies_reasoning['num_mentioned_actors'] = movies_reasoning['mentioned_actors'].apply(len)
    movies_reasoning['mentions_any_actor'] = movies_reasoning['num_mentioned_actors'] > 0

    # The movie description does not mention any actor's real name
    movies_reasoning['does_not_mention_any_actor'] = movies_reasoning['num_mentioned_actors'] == 0

    movies_reasoning.head(100).to_csv(f"{output_dir}/movies_reasoning_actor_mentions.csv", index=False)

    # The lead actor is mentioned first
    movies_reasoning = movies_reasoning[movies_reasoning['num_mentioned_actors'] > 0]
    movies_reasoning['lead_actor'] = movies_reasoning['actors'].str.split(", ").str[0]
    movies_reasoning['lead_actor_first'] = movies_reasoning['lead_actor'] == movies_reasoning['mentioned_actors'].str[0]

    movies_reasoning[['rotten_tomatoes_link', 'movie_title', 'movie_info', 'actors', 'lead_actor', 'lead_actor_first']].head(100).to_csv(f"{output_dir}/movies_reasoning_lead_actor_first.csv", index=False)

def create_reviews_dataset(drop_subset_reviews: List[str] = ['review_content'], drop_subset_movies: List[str] = ['movie_title'], input_dir="datasets/rotten_tomatoes", output_dir="datasets/rotten_tomatoes/quality_exps"):
    reviews = pd.read_csv(f"{input_dir}/reviews.csv").dropna(subset=drop_subset_reviews)
    movies = pd.read_csv(f"{input_dir}/movies.csv").dropna(subset=drop_subset_movies)

    ### Derivation Curation
    movies_reviews = movies.merge(reviews, on='rotten_tomatoes_link')[['rotten_tomatoes_link', 'movie_title', 'review_content', 'review_type']]

    # The review text contains the title of the corresponding movie
    movies_reviews['title_in_review'] = movies_reviews.apply(lambda row: contains(row, 'movie_title', 'review_content', threshold=85), axis=1)
    movies_reviews = movies_reviews[movies_reviews['title_in_review'] == True].drop_duplicates(subset=['review_content'])

    movies_reviews[['rotten_tomatoes_link', 'review_content', 'movie_title', 'review_type']].head(100).to_csv(f"{output_dir}/reviews_derivation_curated.csv", index=False)

    ### Selection Curation
    # Does the review recommends the movie?
    reviews['is_fresh'] = reviews['review_type'].str.lower() == 'fresh'
    reviews[['review_content', 'review_type', 'is_fresh']].head(100).to_csv(f"{output_dir}/reviews_selection_curated.csv", index=False)

def create_join_tables(drop_subset: List[str] = ['movie_info'], input_dir="datasets/rotten_tomatoes", output_dir="datasets/rotten_tomatoes/quality_exps/join_tables"):
    os.makedirs(output_dir, exist_ok=True)

    # Movie description mentions actor
    os.makedirs(f"{output_dir}/description_mentions_actor", exist_ok=True)

    movies = pd.read_csv(f"{input_dir}/movies.csv").dropna(subset=drop_subset)
    movies['mentioned_actors'] = movies['movie_info'].str.findall(r'\((.*?)\)').apply(lambda groups:[name.strip() for g in groups for name in g.split(",")])

    gt_df = movies[['rotten_tomatoes_link', 'movie_title', 'mentioned_actors']].explode('mentioned_actors').dropna().rename(columns={'mentioned_actors': 'actor'})
    
    movie_descriptions = movies[['rotten_tomatoes_link', 'movie_title', 'movie_info']].head(10)
    actors = gt_df[['actor']].head(10)

    movie_descriptions.to_csv(f"{output_dir}/description_mentions_actor/movie_descriptions.csv", index=False)
    actors.to_csv(f"{output_dir}/description_mentions_actor/actors.csv", index=False)

    cross_df = movie_descriptions.merge(actors, how='cross')
    results_df = cross_df.merge(gt_df, on=['rotten_tomatoes_link', 'movie_title', 'actor'], how='inner')
    results_df[['rotten_tomatoes_link', 'movie_title', 'movie_info', 'actor']].to_csv(f"{output_dir}/description_mentions_actor/ground_truth.csv", index=False)

    # Review praises an actor
    os.makedirs(f"{output_dir}/review_praises_actor", exist_ok=True)
    reviews = pd.read_csv("datasets/rotten_tomatoes/reviews.csv")[['rotten_tomatoes_link', 'critic_name', 'review_content']].dropna().drop_duplicates()

    movies = pd.read_csv(f"datasets/rotten_tomatoes/movies.csv").dropna(subset=['movie_info'])[['rotten_tomatoes_link', 'movie_title', 'movie_info']]
    movies['mentioned_actors'] = movies['movie_info'].str.findall(r'\((.*?)\)').apply(lambda groups:[name.strip() for g in groups for name in g.split(",")])

    movies_actors = movies.explode('mentioned_actors').dropna(subset=['mentioned_actors']).rename(columns={'mentioned_actors': 'actor'})

    reviews_movies_actors = movies_actors.merge(reviews, on=['rotten_tomatoes_link'])
    reviews_movies_actors[['rotten_tomatoes_link', 'movie_title', 'review_content', 'critic_name', 'actor']]

    praise_constraints = [('Percy Jackson & the Olympians: The Lightning Thief', 'Drew McWeeny', 'Logan Lerman'), ('Please Give', 'Tim Grierson', 'Catherine Keener'), ('Please Give', 'Steve Ramos', 'Catherine Keener'), ('12 Angry Men (Twelve Angry Men)', 'M. Faust', 'Henry Fonda'), ("20,000 Leagues Under The Sea", "Christopher Lloyd", "James Mason"), ('The 39 Steps', 'Daniel Etherington', 'Robert Donat'), ('The 39 Steps', 'Daniel Etherington', 'Madeleine Carroll'), ('3:10 to Yuma', 'Emanuel Levy', 'Glenn Ford'), ('3:10 to Yuma', 'Emanuel Levy', 'Van Heflin'), ('Abraham Lincoln', 'Phil Hall', 'Walter Huston')]

    constraint_df = pd.DataFrame(praise_constraints, columns=['movie_title', 'critic_name', 'actor'])
    result_df = reviews_movies_actors.merge(constraint_df, on=['movie_title', 'critic_name', 'actor'], how='inner')

    result_df[['actor']].to_csv(f"{output_dir}/review_praises_actor/actors.csv", index=False)
    reviews = result_df[['review_content']].drop_duplicates()

    movie_10000_BC = reviews_movies_actors[reviews_movies_actors['movie_title'] == '10,000 B.C.'].iloc[[0]][['review_content']]
    movie_dark_water = reviews_movies_actors[reviews_movies_actors['movie_title'] == 'Dark Water'].iloc[[0]][['review_content']]

    reviews = pd.concat([reviews, movie_10000_BC, movie_dark_water], ignore_index=True)
    reviews.to_csv(f"{output_dir}/review_praises_actor/reviews.csv", index=False)

    result_df[['rotten_tomatoes_link', 'movie_title', 'review_content', 'actor']].to_csv(f"{output_dir}/review_praises_actor/ground_truth.csv")

    # Review critisize movie
    os.makedirs(f"{output_dir}/review_critisize_movie", exist_ok=True)
    reviews = pd.read_csv("datasets/rotten_tomatoes/reviews.csv")[['rotten_tomatoes_link', 'review_content']].dropna(subset=['review_content']).drop_duplicates(subset=['review_content'])
    movies = pd.read_csv(f"datasets/rotten_tomatoes/movies.csv").dropna(subset=['movie_info'])[['rotten_tomatoes_link', 'movie_title']].dropna(subset=['movie_title']).drop_duplicates(subset=['movie_title'])

    reviews_movies = movies.merge(reviews, on=['rotten_tomatoes_link']).head(1000)

    reviews_movies['title_in_review'] = reviews_movies.apply(lambda row: contains(row, 'movie_title', 'review_content'), axis=1)
    reviews_movies = reviews_movies[reviews_movies['title_in_review'] == True].drop_duplicates(subset=['movie_title']).head(100)

    reviews = reviews_movies[['review_content']].head(10)
    movies = reviews_movies[['movie_title']].head(10)

    reviews.to_csv(f"{output_dir}/review_critisize_movie/reviews.csv", index=False)
    movies.to_csv(f"{output_dir}/review_critisize_movie/movies.csv", index=False)
    reviews_movies.to_csv(f"{output_dir}/review_critisize_movie/ground_truth.csv", index=False)

    # Both movies same {genre, target audience}
    os.makedirs(f"{output_dir}/movies_same_genre_audience", exist_ok=True)

    movies = pd.read_csv("datasets/rotten_tomatoes/movies.csv")[['rotten_tomatoes_link', 'movie_title', 'movie_info', 'genres', 'content_rating']].dropna(subset=['movie_info', 'genres', 'content_rating'])

    movies = movies[movies["genres"].str.split().str.len() == 1]
    movies = movies[movies["content_rating"].isin(['G', 'PG', 'R'])]

    m1 = movies.copy().head(10).rename(columns={'movie_title': 'movie_title_1', 'movie_info': 'movie_info_1'})
    m2 = movies.copy().tail(10).rename(columns={'movie_title': 'movie_title_2', 'movie_info': 'movie_info_2'})

    gt_df = m1.merge(m2, on=['genres', 'content_rating'])

    m1[['rotten_tomatoes_link', 'movie_title_1', 'movie_info_1']].to_csv(f"{output_dir}/movies_same_genre_audience/movies_1.csv", index=False)
    m2[['rotten_tomatoes_link', 'movie_title_2', 'movie_info_2']].to_csv(f"{output_dir}/movies_same_genre_audience/movies_2.csv", index=False)
    gt_df[['movie_title_1', 'movie_title_2', 'movie_info_1', 'movie_info_2']].to_csv(f"{output_dir}/movies_same_genre_audience/ground_truth.csv", index=False)
    
    # Both reviews recommend the movie
    os.makedirs(f"{output_dir}/fresh_reviews", exist_ok=True)

    reviews = pd.read_csv("datasets/rotten_tomatoes/reviews.csv")[['rotten_tomatoes_link', 'review_type', 'review_content']].dropna(subset=['review_content']).drop_duplicates(subset=['review_content'])
    
    reviews_1 = reviews.copy().head(10).rename(columns={'review_content': 'review_content_1'})
    reviews_2 = reviews.copy().tail(10).rename(columns={'review_content': 'review_content_2'})

    fresh_reviews = reviews_1.merge(reviews_2, on=['review_type'])
    fresh_reviews = fresh_reviews[fresh_reviews['review_type'] == 'Fresh']

    reviews_1[['review_content_1']].to_csv(f"{output_dir}/fresh_reviews/reviews_1.csv", index=False)
    reviews_2[['review_content_2']].to_csv(f"{output_dir}/fresh_reviews/reviews_2.csv", index=False)
    fresh_reviews[['review_content_1', 'review_content_2']].to_csv(f"{output_dir}/fresh_reviews/ground_truth.csv", index=False)

    # Movies-Directors
    os.makedirs(f"{output_dir}/movies_directors", exist_ok=True)

    movies = pd.read_csv("datasets/rotten_tomatoes/movies.csv")[['rotten_tomatoes_link', 'movie_title', 'directors']].dropna(subset=['movie_title', 'directors'])

    # Keep movies with exactly one director
    movies = movies[movies["directors"].str.split(", ").str.len() == 1].rename(columns={'directors': 'director'})

    movies[['movie_title']].to_csv(f"{output_dir}/movies_directors/movies.csv", index=False)
    movies[['director']].to_csv(f"{output_dir}/movies_directors/directors.csv", index=False)
    movies[['movie_title', 'director']].head(10).to_csv(f"{output_dir}/movies_directors/ground_truth.csv", index=False)

   # Movies-Authors
    os.makedirs(f"{output_dir}/movies_authors", exist_ok=True)

    movies = pd.read_csv("datasets/rotten_tomatoes/movies.csv")[['rotten_tomatoes_link', 'movie_title', 'authors']].dropna(subset=['movie_title', 'authors'])

    # Keep movies with exactly one author
    movies = movies[movies["authors"].str.split(", ").str.len() == 1].rename(columns={'authors': 'author'})

    movies[['movie_title']].to_csv(f"{output_dir}/movies_authors/movies.csv", index=False)
    movies[['author']].to_csv(f"{output_dir}/movies_authors/authors.csv", index=False)
    movies[['movie_title', 'author']].head(10).to_csv(f"{output_dir}/movies_authors/ground_truth.csv", index=False)

    # Prequel-Sequel relationship
    os.makedirs(f"{output_dir}/prequel_sequel", exist_ok=True)

    movies = pd.read_csv("datasets/rotten_tomatoes/movies.csv")[['rotten_tomatoes_link', 'movie_title', "original_release_date"]]

    prequels = [
        "21 Jump Street",
        "28 Days Later",
        "30 Days of Night",
        "Airplane!",
        "Creep",
        "Analyze This",
        "Arthur",
        "Crank",
        "Happy Death Day",
        "Harold & Kumar",
    ]

    sequels = [
        "22 Jump Street",
        "28 Weeks Later...",
        "30 Days Of Night: Dark Days",
        "Airplane 2 - The Sequel",
        "Creep 2",
        "Analyze That",
        "Arthur 2: On the Rocks",
        "Crank 2: High Voltage",
        "Happy Death Day 2U",
        "Harold & Kumar Escape from Guantanamo Bay"
    ]

    prequels_df = pd.DataFrame({'movie_title_1': prequels})
    sequels_df = pd.DataFrame({'movie_title_2': sequels})

    gt_df = pd.concat([prequels_df, sequels_df], axis=1)

    prequels_df.to_csv(f"{output_dir}/prequel_sequel/prequels.csv", index=False)
    sequels_df.to_csv(f"{output_dir}/prequel_sequel/sequels.csv", index=False)
    gt_df.to_csv(f"{output_dir}/prequel_sequel/ground_truth.csv", index=False)





if __name__ == "__main__":
    os.makedirs("datasets/rotten_tomatoes/quality_exps", exist_ok=True)

    # create_movie_dataset()
    create_reviews_dataset()
    # create_join_tables()

    print("All pre-processing tasks completed successfully!")






