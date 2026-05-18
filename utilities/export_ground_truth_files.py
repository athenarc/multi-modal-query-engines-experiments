import pandas as pd

q1 = pd.read_csv("datasets/nba/game_summaries.csv")[['sportsett_id', 'summary', 'win_home_or_vis']].head(100)
q1.to_csv("datasets/nba/ground_truths/q1.csv")

q2 = pd.read_csv("datasets/nba/game_summaries.csv")[['sportsett_id', 'summary', 'winner_points']].head(100)
q2.to_csv("datasets/nba/ground_truths/q2.csv")

q3 = pd.read_csv("datasets/nba/players_summaries.csv")[['sportsett_id', 'player_name', 'summary', 'points']].head(100)
q3.to_csv("datasets/nba/ground_truths/q3.csv")

q4 = pd.read_csv("datasets/nba/all_seasons.csv")[['sportsett_id', 'team', 'summary', 'is_winner']].head(100)