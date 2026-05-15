import pandas as pd
import datasets
import re
import unicodedata

def create_player_summary_dataset(dataset):
    print("Creating player summary dataset...")
    
    csv_data = []
    total_players_evaluated = 0

    print(f"Processing {len(dataset)} games...")
    
    for row in dataset:
        game_id = row.get("sportsett_id")
        
        # Keep the first summary if "target" is not available
        if "target" in row and row["target"]:
            summary = str(row["target"])
        elif "summaries" in row and isinstance(row["summaries"], list) and len(row["summaries"]) > 0:
            summary = str(row["summaries"][0])
        else:
            summary = "No summary available"
            
        teams = row.get("teams", {})
        
        # Loop through both 'home' and 'vis' (visitor) teams
        for team_type in ["home", "vis"]:
            team_info = teams.get(team_type, {})
            team_name = team_info.get("name", "Unknown Team")
            box_score = team_info.get("box_score", [])
            
            # Loop through every player in the box score
            for player in box_score:
                total_players_evaluated += 1
                player_name = str(player.get("name", "Unknown Player"))
                
                # Check if the player's name (or last name) is mentioned in the summary
                is_mentioned = False
                
                if player_name in summary:
                    is_mentioned = True
                else:
                    name_parts = player_name.split(' ', 1)
                    if len(name_parts) > 1:
                        last_name = name_parts[-1]
                        if re.search(r'\b' + re.escape(player_name) + r'\b', summary):
                            is_mentioned = True
                
                # Only extract metrics and append to our data list if the player is mentioned
                if is_mentioned:
                    pts = player.get("PTS", "0")    # Points
                    reb = player.get("TREB", "0")   # Total Rebounds
                    ast = player.get("AST", "0")    # Assists
                    stl = player.get("STL", "0")    # Steals
                    mins = player.get("MIN", "0")   # Minutes Played
                    
                    csv_data.append({
                        "sportsett_id": game_id,
                        "team": team_name,
                        "player_name": player_name,
                        "minutes_played": mins,
                        "points": pts,
                        "rebounds": reb,
                        "assists": ast,
                        "steals": stl,
                        "summary": summary
                    })

    # Convert the list of filtered dictionaries into a pandas DataFrame
    df_players_summaries = align_summaries_to_nba_players(pd.DataFrame(csv_data))
    
    print("-" * 30)
    print("Processing and filtering completed.")
    print(f"Original player rows evaluated: {total_players_evaluated}")
    print(f"Player rows after filtering: {df_players_summaries.shape[0]}")

    return df_players_summaries

def align_summaries_to_nba_players(df_summaries_players):
    seasons = pd.read_csv("datasets/nba/all_seasons.csv", index_col=0)

    # Remove dots from names
    seasons['player_name'] = seasons['player_name'].str.replace('.', '', regex=False)
    df_summaries_players['player_name'] = df_summaries_players['player_name'].str.replace('.', '', regex=False)

    # Remove generation suffixes
    seasons['player_name'] = seasons['player_name'].str.replace(r'\s+(Jr|I|II|III|IV)$', '', regex=True)
    df_summaries_players['summary'] = df_summaries_players['summary'].str.replace(r'\s*(Jr\.|, Jr\.|Jr,|II)', '', regex=True)
    df_summaries_players['summary'] = df_summaries_players['summary'].str.replace(r'\s*( Jr )', ' ', regex=True)


    # Convert all characters to ascii and remove accents
    df_summaries_players['player_name'] = df_summaries_players['player_name'].apply(
        lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8')
    )

    # Handle specific name variations
    df_summaries_players['player_name'] = df_summaries_players['player_name'].replace({
        "Mitch Creek": "Mitchell Creek",
        "Mohamed Bamba": "Mo Bamba",
        "Sviatoslav Mykhailiuk": "Svi Mykhailiuk",
        "Wesley Iwundu": "Wes Iwundu"
    })
    df_summaries_players['summary'] = df_summaries_players['summary'].str.replace(r'\bMitch Creek\b', 'Mitchell Creek', regex=True)
    df_summaries_players['summary'] = df_summaries_players['summary'].str.replace(r'\bMohamed Bamba\b', 'Mo Bamba', regex=True)
    df_summaries_players['summary'] = df_summaries_players['summary'].str.replace(r'\bSviatoslav Mykhailiuk\b', 'Svi Mykhailiuk', regex=True)
    df_summaries_players['summary'] = df_summaries_players['summary'].str.replace(r'\bWesley Iwundu\b', 'Wes Iwundu', regex=True)

    for player in df_summaries_players['player_name'].unique().tolist():
        if player not in seasons['player_name'].unique().tolist():
            print("-" * 30)
            print(f"Warning: Player '{player}' from SportSett is not in the Kaggle dataset.")
            print()

    seasons.to_csv("datasets/nba/all_seasons_cleaned.csv", index=False, encoding='utf-8')
    df_summaries_players.to_csv("datasets/nba/players_summaries.csv", index=False, encoding='utf-8')
    
    print("-" * 30)
    print("Dataset alignment completed.\nCleaned datasets saved as all_seasons_cleaned.csv and players_summaries.csv under datasets/nba/ directory.")

    return df_summaries_players

def create_team_summary_dataset(dataset):
    print("Creating team summary dataset...")

    csv_data = []

    print(f"Processing {len(dataset)} games...")

    for row in dataset:
        game_id = row.get("sportsett_id")
        
        # Keep the first summary if "target" is not available
        if "target" in row and row["target"]:
            summary = str(row["target"])
        elif "summaries" in row and isinstance(row["summaries"], list) and len(row["summaries"]) > 0:
            summary = str(row["summaries"][0])
        else:
            summary = "No summary available"
            
        teams = row.get("teams", {})

        home = (row['teams']['home']['name'], int(row['teams']['home']['line_score']['game']['PTS']))
        vis = (row['teams']['vis']['name'], int(row['teams']['vis']['line_score']['game']['PTS']))

        winner_team = home[0] if home[1] > vis[1] else vis[0]

        csv_data.append({
            "sportsett_id": game_id,
            "team": home[0],
            "summary": summary,
            "is_winner": home[0] == winner_team
        })
        csv_data.append({
            "sportsett_id": game_id,
            "team": vis[0],
            "summary": summary,
            "is_winner": vis[0] == winner_team
        })

        #TODO: Check if the summary contains the winner team name.

    pd.DataFrame(csv_data).to_csv("datasets/nba/teams_summaries.csv", index=False, encoding='utf-8')
    print("-" * 30)
    print("Team-Summary alignment completed.\nDataset saved as teams_summaries.csv under datasets/nba/ directory.")


def create_game_summaries_df(dataset):
    game_summaries = []
    
    for row in dataset:
        game_id = row.get("sportsett_id")

        if "target" in row and row["target"]:
            summary = str(row["target"])
        elif "summaries" in row and isinstance(row["summaries"], list) and len(row["summaries"]) > 0:
            summary = str(row["summaries"][0])
        else:
            summary = "No summary available"

        home = (row['teams']['home']['name'], int(row['teams']['home']['line_score']['game']['PTS']))
        vis = (row['teams']['vis']['name'], int(row['teams']['vis']['line_score']['game']['PTS']))

        if home[1] > vis[1]:
            winner_team = home[0]
            winner_points = home[1]
        else:
            winner_team = vis[0]
            winner_points = vis[1]
        total_points = home[1] + vis[1]


        #TODO: Check if the summary contains the winner team name, points, etc.


        game_summaries.append({
            "sportsett_id": game_id,
            "winner_team": winner_team,
            "winner_points": winner_points,
            "total_points": total_points,
            "summary": summary
        })
    
    pd.DataFrame(game_summaries).to_csv("datasets/nba/game_summaries.csv", index=False, encoding='utf-8')
    print("-" * 30)
    print("Game summaries extracted and saved as game_summaries.csv under datasets/nba/ directory.")

if __name__ == "__main__":
    dataset = datasets.load_dataset(
        "parquet", 
        data_files={
            "train": "https://huggingface.co/datasets/GEM/sportsett_basketball/resolve/refs/convert/parquet/default/train/*.parquet", 
            "validation": "https://huggingface.co/datasets/GEM/sportsett_basketball/resolve/refs/convert/parquet/default/validation/*.parquet", 
            "test": "https://huggingface.co/datasets/GEM/sportsett_basketball/resolve/refs/convert/parquet/default/test/*.parquet"
        }
    )
    
    create_player_summary_dataset(dataset['test'])
    # create_game_summaries_df(dataset['test'])
    # create_team_summary_dataset(dataset['test'])