import pandas as pd
import datasets
import re
import unicodedata

def process_and_filter_sportssett(dataset):
    print("Loading dataset...")
    
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
                
                # --- FILTERING LOGIC ---
                is_mentioned = False
                
                # 1. Check if their exact full name is in the summary
                if player_name in summary:
                    is_mentioned = True
                else:
                    # 2. Check last name only (if the player has more than one name part)
                    name_parts = player_name.split(' ', 1)
                    if len(name_parts) > 1:
                        last_name = name_parts[-1]
                        if re.search(r'\b' + re.escape(last_name) + r'\b', summary):
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
    df = pd.DataFrame(csv_data)
    
    print("-" * 30)
    print("Processing and filtering completed.")
    print(f"Original player rows evaluated: {total_players_evaluated}")
    print(f"Player rows after filtering: {len(df)}")

    return df

def align_summaries_to_nba_players(df_summaries_players):
    seasons = pd.read_csv("datasets/nba/all_seasons.csv", index_col=0)

    # Remove dots from names
    seasons['player_name'] = seasons['player_name'].str.replace('.', '', regex=False)
    df_summaries_players['player_name'] = df_summaries_players['player_name'].str.replace('.', '', regex=False)

    # Remove generation suffixes
    seasons['player_name'] = seasons['player_name'].str.replace(r'\s+(Jr|I|II|III|IV)$', '', regex=True)

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

    for player in df_summaries_players['player_name'].unique().tolist():
        if player not in seasons['player_name'].unique().tolist():
            print("-" * 30)
            print(f"Warning: Player '{player}' from SportSett is not in the Kaggle dataset.")
            print()

    seasons.to_csv("datasets/nba/all_seasons_cleaned.csv", index=False, encoding='utf-8')
    df_summaries_players.to_csv("datasets/nba/players_summaries.csv", index=False, encoding='utf-8')
    
    print("-" * 30)
    print("Dataset alignment completed.\nCleaned datasets saved as all_seasons_cleaned.csv and players_summaries.csv under datasets/nba/ directory.")

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
    
    # Notice we only need to call the merged function now
    filtered_summaries = process_and_filter_sportssett(dataset['test'])
    
    align_summaries_to_nba_players(filtered_summaries)

    create_game_summaries_df(dataset['test'])