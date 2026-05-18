import pandas as pd

q1 = pd.read_csv("datasets/nba/game_summaries.csv")[['sportsett_id', 'summary', 'win_home_or_vis']].head(100)
q1.to_csv("datasets/nba/ground_truths/q1.csv", index=False)

q2 = pd.read_csv("datasets/nba/game_summaries.csv")[['sportsett_id', 'summary', 'winner_points']].head(100)
q2.to_csv("datasets/nba/ground_truths/q2.csv", index=False)

q3 = pd.read_csv("datasets/nba/players_summaries.csv")[['sportsett_id', 'player_name', 'summary', 'points']].head(100)
q3.to_csv("datasets/nba/ground_truths/q3.csv", index=False)

q4 = pd.read_csv("datasets/nba/all_seasons.csv")[['player_name', 'country']].head(100)
q4.to_csv("datasets/nba/ground_truths/q4.csv", index=False)

q5 = pd.read_csv("datasets/nba/teams_summaries.csv")[['sportsett_id', 'team', 'summary', 'is_winner']].head(100)
q5.to_csv("datasets/nba/ground_truths/q5.csv", index=False)

q6 = pd.read_csv("datasets/nba/game_summaries.csv")[['sportsett_id', 'summary', 'total_points']].head(100)
q6.to_csv("datasets/nba/ground_truths/q6.csv", index=False)
