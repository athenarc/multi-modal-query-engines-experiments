import os
import unicodedata
import pandas as pd
import datasets
from openai import OpenAI
from tqdm import tqdm

tqdm.pandas(desc="Verification")

client = OpenAI(base_url="http://localhost:5001/v1", api_key="EMPTY")

print("Loading HuggingFace SportSett dataset...")
sportsett_dataset = datasets.load_dataset(
    "parquet", 
    data_files={
        "train": "https://huggingface.co/datasets/GEM/sportsett_basketball/resolve/refs/convert/parquet/default/train/*.parquet", 
        "validation": "https://huggingface.co/datasets/GEM/sportsett_basketball/resolve/refs/convert/parquet/default/validation/*.parquet", 
        "test": "https://huggingface.co/datasets/GEM/sportsett_basketball/resolve/refs/convert/parquet/default/test/*.parquet"
    }
)
test_data = sportsett_dataset['test']

# Constants
NBA_TEAMS = {
    'HOU': 'Houston Rockets', 'DEN': 'Denver Nuggets', 'OKC': 'Oklahoma City Thunder',
    'MIA': 'Miami Heat', 'WAS': 'Washington Wizards', 'LAL': 'Los Angeles Lakers',
    'SAS': 'San Antonio Spurs', 'MEM': 'Memphis Grizzlies', 'NOP': 'New Orleans Pelicans',
    'PHI': 'Philadelphia 76ers', 'IND': 'Indiana Pacers', 'NYK': 'New York Knicks',
    'UTA': 'Utah Jazz', 'ORL': 'Orlando Magic', 'ATL': 'Atlanta Hawks', 'CHI': 'Chicago Bulls',
    'TOR': 'Toronto Raptors', 'LAC': 'Los Angeles Clippers', 'MIL': 'Milwaukee Bucks',
    'CLE': 'Cleveland Cavaliers', 'DET': 'Detroit Pistons', 'PHX': 'Phoenix Suns',
    'DAL': 'Dallas Mavericks', 'CHA': 'Charlotte Hornets', 'BKN': 'Brooklyn Nets',
    'GSW': 'Golden State Warriors', 'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings',
    'BOS': 'Boston Celtics', 'MIN': 'Minnesota Timberwolves'
}

COUNTRY_TO_CONTINENT = {
    'USA': 'America', 'Nigeria': 'Africa', 'Congo': 'Africa', 'Canada': 'America',
    'Serbia and Montenegro': 'Europe', 'Ukraine': 'Europe', 'Croatia': 'Europe',
    'Jamaica': 'America', 'Lithuania': 'Europe', 'Slovenia': 'Europe',
    'US Virgin Islands': 'America', 'France': 'Europe', 'St. Vincent & Grenadines': 'America',
    'Germany': 'Europe', 'Dominican Republic': 'America', 'New Zealand': 'Oceania',
    'Georgia': 'Europe', 'Belize': 'America', 'England': 'Europe', 'Turkey': 'Asia/Europe',
    'Greece': 'Europe', 'Finland': 'Europe', 'Senegal': 'Africa', 'Mexico': 'America',
    'Puerto Rico': 'America', 'China': 'Asia', 'Argentina': 'America', 'Mali': 'Africa',
    'U.S. Virgin Islands': 'America', 'Yugoslavia': 'Europe', 'Spain': 'Europe',
    'Venezuela': 'America', 'Serbia': 'Europe', 'Haiti': 'America', 'Russia': 'Asia/Europe',
    'Brazil': 'America', 'Ireland': 'Europe', 'Scotland': 'Europe', 'Poland': 'Europe',
    'Netherlands': 'Europe', 'Czech Republic': 'Europe', 'Montenegro': 'Europe',
    'United Kingdom': 'Europe', 'Democratic Republic of the Congo': 'Africa', 'Latvia': 'Europe',
    'South Korea': 'Asia', 'USSR': 'Europe', 'Australia': 'Oceania', 'Uruguay': 'America',
    'Sudan (UK)': 'Africa', 'Italy': 'Europe', 'Switzerland': 'Europe', 'Gabon': 'Africa',
    'Cameroon': 'Africa', 'Iran': 'Asia', 'Israel': 'Asia', 'Tanzania': 'Africa',
    'Sweden': 'Europe', 'Panama': 'America', 'Great Britain': 'Europe', 'Bosnia': 'Europe',
    'Macedonia': 'Europe', 'Bosnia & Herzegovina': 'Europe', 'Cabo Verde': 'Africa',
    'Tunisia': 'Africa', 'South Sudan': 'Africa', 'Bahamas': 'America', 'Ghana': 'Africa',
    'Austria': 'Europe', 'Bosnia and Herzegovina': 'Europe', 'Egypt': 'Africa',
    'Trinidad and Tobago': 'America', 'Japan': 'Asia', 'Angola': 'Africa',
    'Saint Lucia': 'America', 'Sudan': 'Africa', 'DRC': 'Africa',
    'Republic of the Congo': 'Africa', 'Guinea': 'Africa', 'Denmark': 'Europe',
    'Colombia': 'America', 'Portugal': 'Europe'
}

# Helpers
def get_summary(row):
    if "target" in row and row["target"]:
        return str(row["target"])
    elif "summaries" in row and isinstance(row["summaries"], list) and len(row["summaries"]) > 0:
        return str(row["summaries"][0])
    return "No summary available"

def llm_verify(prompt, model="RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Error during LLM verification: {e}")
        return "no"

# Core Pre-Processing Methods
def process_player_info(input_csv="datasets/nba/all_seasons.csv", output_csv="datasets/nba/quality_exps/players_info.csv"):
    print("Processing Player Info...")
    stats = pd.read_csv(input_csv, index_col=0)
    stats = stats[stats['season'] == '2021-22']
    stats['team_name'] = stats['team_abbreviation'].map(NBA_TEAMS)
    
    stats = stats[~stats['draft_year'].isin(['Undrafted'])]
    stats = stats[~stats['draft_round'].isin(['Undrafted', '0'])]
    
    stats = stats.head(100)[['player_name', 'team_name', 'player_height', 'country', 'draft_year', 'draft_round', 'college', 'age', 'season']]
    
    stats['country_continent'] = stats['country'].map(COUNTRY_TO_CONTINENT)
    stats['born_in_america'] = stats['country_continent'] == 'America'
    stats['height_lt_200'] = stats['player_height'] < 200
    stats['first_round_drafted'] = stats['draft_round'] == '1'
    
    stats.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

def process_player_summaries(output_csv="datasets/nba/quality_exps/players_summaries_stats.csv"):
    print("Processing Player Summaries...")
    csv_data = []
    
    for row in test_data.select(range(40)):
        game_id = row.get("sportsett_id")
        summary = get_summary(row)
        
        for team_type in ["home", "vis"]:
            team_info = row.get("teams", {}).get(team_type, {})
            team_name = team_info.get("name", "Unknown Team")
            
            for player in team_info.get("box_score", []):
                player_name = str(player.get("name", "Unknown Player"))
                pts, ast, reb = player.get("PTS", "0"), player.get("AST", "0"), player.get("TREB", "0")
                
                is_mentioned = (player_name in summary) or (unicodedata.normalize('NFKD', player_name).encode('ASCII', 'ignore').decode('utf-8') in summary)
                player_name = unicodedata.normalize('NFKD', player_name).encode('ASCII', 'ignore').decode('utf-8')         

                csv_data.append({
                    "sportsett_id": game_id, "summary": summary, "player_name": player_name,
                    "team": team_name, "points": int(pts), "assists": int(ast), "total_rebounds": int(reb),
                    "points_gt_20": int(pts) > 20, "is_mentioned": is_mentioned
                })

    df = pd.DataFrame(csv_data)
    df[['sportsett_id', 'player_name', 'summary', 'is_mentioned']].to_csv("datasets/nba/quality_exps/players_summaries_all.csv", index=False)
    
    df = df[df['is_mentioned'] == True].copy()

    def verify_stat(row, stat_name, stat_val):
        prompt = f"""
        Task: Check if a basketball summary mentions the {stat_name} by a specific player.
        Player: {row['player_name']}
        {stat_name.capitalize()}: {stat_val}
        Summary: "{row['summary']}"
        Question: Does the summary explicitly mention that the player {row['player_name']} had {stat_val} {stat_name}?
        Answer ONLY with 'Yes' or 'No'. Do not include any other words or punctuation.
        """
        return llm_verify(prompt)

    df['points_verified'] = df.progress_apply(lambda r: verify_stat(r, 'points', r['points']), axis=1)
    df = df[df['points_verified'] == 'yes']

    df['assists_verified'] = df.progress_apply(lambda r: verify_stat(r, 'assists', r['assists']), axis=1)
    df = df[df['assists_verified'] == 'yes']

    df['rebounds_verified'] = df.progress_apply(lambda r: verify_stat(r, 'total rebounds', r['total_rebounds']), axis=1)
    df = df[df['rebounds_verified'] == 'yes']

    verified_summaries = df[df['points'] != 0]
    verified_summaries = verified_summaries[verified_summaries['assists'] != 0]
    verified_summaries = verified_summaries[verified_summaries['total_rebounds'] != 0]

    verified_summaries.head(100).drop(columns=['points_verified', 'assists_verified', 'rebounds_verified']).reset_index(drop=True).to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

def process_game_summaries(output_csv="datasets/nba/quality_exps/game_summaries.csv"):
    print("Processing Game Summaries...")
    game_summaries = []
    
    for row in test_data.select(range(200)):
        game_id = row.get("sportsett_id")
        summary = get_summary(row)

        home = row['teams']['home']
        vis = row['teams']['vis']
        
        home_pts = int(home['line_score']['game']['PTS'])
        vis_pts = int(vis['line_score']['game']['PTS'])

        if home_pts > vis_pts:
            winner, loser = (home['name'], home_pts, 'home', home['next_game']['opponent_name']), (vis['name'], vis_pts, 'visitor')
        else:
            winner, loser = (vis['name'], vis_pts, 'visitor', vis['next_game']['opponent_name']), (home['name'], home_pts, 'home')

        total_points = home_pts + vis_pts
        
        game_summaries.append({
            'sportsett_id': game_id, "summary": summary,
            "winner_team": winner[0], "winner_points": winner[1],
            "loser_team": loser[0], "loser_points": loser[1],
            "win_home_or_vis": winner[2], "loss_home_or_vis": loser[2],
            "winner_next_opponent": winner[3],   
            "def_or_off": "defensive" if total_points < 210 else "offensive",
            "double_digit_margin": (10 <= abs(home_pts - vis_pts) < 100),
            "total_points_gt_180": total_points > 180
        })

    df = pd.DataFrame(game_summaries)

    def verify_opponent(row):
        prompt = f"""
        Task: Check if a sports summary mentions the next opponent of the winner team.
        Summary: "{row['summary']}"
        Question: Does the summary explicitly mention the next opponent of the winner team?
        Answer ONLY with 'Yes' or 'No'. Do not include any other words or punctuation.
        """
        return llm_verify(prompt)

    df['winner_opponent_mentioned'] = df.progress_apply(verify_opponent, axis=1)
    
    verified = df[df['winner_opponent_mentioned'] == 'yes'].head(100).drop(columns=['winner_opponent_mentioned']).reset_index(drop=True)
    verified.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

def process_game_summaries_all(output_csv: str = "datasets/nba/quality_exps/game_summaries_all.csv"):
    game_summaries = []

    for row in test_data:
        game_id = row.get("sportsett_id")
        summary = get_summary(row)

        game_summaries.append({
            'sportsett_id': game_id, "summary": summary
        })

    all_summaries = pd.DataFrame(game_summaries)
    all_summaries.to_csv(output_csv, index=False)
    print(f"All summaries saved to {output_csv}")

def process_team_summaries(output_csv="datasets/nba/quality_exps/teams_summaries.csv", output_csv_all="datasets/nba/quality_exps/teams_summaries_all.csv"):
    print("Processing Team Summaries...")
    csv_data = []

    for row in test_data.select(range(50)):
        game_id = row.get("sportsett_id")
        summary = get_summary(row)

        home_name = row['teams']['home']['name']
        home_pts = int(row['teams']['home']['line_score']['game']['PTS'])
        vis_name = row['teams']['vis']['name']
        vis_pts = int(row['teams']['vis']['line_score']['game']['PTS'])

        winner_team = home_name if home_pts > vis_pts else vis_name

        for team in [home_name, vis_name]:
            csv_data.append({
                "sportsett_id": game_id,
                "team": team,
                "summary": summary,
                "is_winner": team == winner_team,
                "is_loser": team != winner_team
            })

    pd.DataFrame(csv_data).to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Saved to {output_csv}")

    csv_data = []

    for row in test_data.select(range(50)):
        game_id = row.get("sportsett_id")
        summary = get_summary(row)

        home_name = row['teams']['home']['name']
        vis_name = row['teams']['vis']['name']
        
        home_next_opponent = row['teams']['home']['next_game']['opponent_name']
        vis_next_opponent = row['teams']['vis']['next_game']['opponent_name']

        for team in [home_name, vis_name, home_next_opponent, vis_next_opponent]:
            if team is not None and team in summary:
                csv_data.append({
                    "sportsett_id": game_id,
                    "team": team,
                    "summary": summary
                })
    pd.DataFrame(csv_data).to_csv(output_csv_all, index=False)   
    print(f"Saved to {output_csv_all}") 
    

def evaluate_joins(title, gt_df, cases, merge_keys, left_table_name: str, right_table_name: str, summaries_df, filter_query=None):
    query_folder = f"datasets/nba/quality_exps/quality_exps/join_tables/{title}"
    os.makedirs(query_folder, exist_ok=True)

    expected_joins = [5, 20, 50]

    print(f"--- {title} ---")
    for i, case_data in enumerate(cases, 1):
        keys = list(case_data.keys())
        
        df1 = pd.DataFrame({keys[0]: case_data[keys[0]]})
        df2 = pd.DataFrame({keys[1]: case_data[keys[1]]})
        
        cross_df = df1.merge(df2, how='cross')
        
        result_df = cross_df.merge(gt_df, on=merge_keys, how='inner')
        
        if filter_query:
            result_df = result_df.query(filter_query)
            
        print(f"Case {i}: {result_df.shape[0]} joins.")

        if result_df.shape[0] == expected_joins[i-1]:
            cpath = os.path.join(query_folder, f"case_{i}")
            os.makedirs(cpath, exist_ok=True)

            lpath = os.path.join(cpath, f"{left_table_name}.csv")
            rpath = os.path.join(cpath, f"{right_table_name}.csv")

            if "sportsett_id" in result_df.columns:
                result_df = result_df.merge(summaries_df, on='sportsett_id')

            if left_table_name == "summaries" and "sportsett_id" in df1.columns:
                df1 = df1.merge(summaries_df, on='sportsett_id')
            elif right_table_name == "summaries" and 'sportsett_id' in df2.columns:
                df2 = df2.merge(summaries_df, on='sportsett_id')

            df1.to_csv(lpath)
            df2.to_csv(rpath)
            result_df.to_csv(os.path.join(cpath, "ground_truth.csv"))
            
    print()

def evaluate_all_joins():
    summaries_df = pd.read_csv("datasets/nba/quality_exps/game_summaries_all.csv")[['sportsett_id', 'summary']]

    # ---------------------------------------------------------
    # 1. Summary mentions Player 
    # ---------------------------------------------------------
    gt_df_1 = pd.read_csv("datasets/nba/quality_exps/players_summaries_all.csv")[['sportsett_id', 'player_name', "is_mentioned"]]

    c1_1 = {
        'sportsett_id': [4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930],
        'player_name': ["Robert Covington", "Furkan Korkmaz", "Jonah Bolden", "Shake Milton", "Demetrius Jackson", "Rawle Alkins", "Melvin Frazier", "Isaiah Briscoe", "Justin Holiday", "Cameron Payne"]
    }
    c1_2 = {
        'sportsett_id': [4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930],
        'player_name': ["Joel Embiid", "Ben Simmons", "Amir Johnson", "Furkan Korkmaz", "Jonah Bolden", "Shake Milton", "Demetrius Jackson", "Rawle Alkins", "Melvin Frazier", "Isaiah Briscoe"]
    }
    c1_3 = {
        'sportsett_id': [4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930, 4931, 4933],
        'player_name': ["Joel Embiid", "Ben Simmons", "J.J. Redick", "Jimmy Butler", "Robert Covington", "Markelle Fultz", "Mike Muscala", "Kemba Walker", "Jeremy Lamb", "Cody Zeller"]
    }
    evaluate_joins("summary_mentions_player", gt_df_1, [c1_1, c1_2, c1_3], ['sportsett_id', 'player_name'], "summaries", "players", summaries_df, "is_mentioned == True")

    # ---------------------------------------------------------
    # 2. Summary mentions Team
    # ---------------------------------------------------------
    gt_df_2 = pd.read_csv("datasets/nba/quality_exps/teams_summaries_all.csv")[['sportsett_id', 'team']].rename(columns={'team': 'team_name'})

    c2_1 = {'sportsett_id': [4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930],
    'team_name': ["Pistons", "Warriors", "Kings", "Nuggets", "Thunder", "Trail Blazers", "Timberwolves", "Pacers", "Rockets", "Mavericks"]}

    c2_2 = {'sportsett_id': [4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930],
    'team_name': ["76ers", "Pistons", "Bulls", "Magic", "Clippers", "Warriors", "Kings", "Nuggets", "Thunder", "Trail Blazers"]}

    c2_3 = {'sportsett_id': [4941, 4942, 4943, 4944, 4945, 4946, 4947, 4948, 4949, 4950],
    'team_name': ["76ers", "76ers", "76ers", "76ers", "Knicks", "Lakers", "Bulls", "Magic", "Clippers", "Hornets"]}
    evaluate_joins("summary_mentions_team", gt_df_2, [c2_1, c2_2, c2_3], ['sportsett_id', 'team_name'], "summaries", "teams", summaries_df)

    # ---------------------------------------------------------
    # 3. Team won the game
    # ---------------------------------------------------------
    gt_df_3 = pd.read_csv("datasets/nba/quality_exps/teams_summaries.csv")[['sportsett_id', 'team', 'is_winner']].rename(columns={'team': 'team_name'})

    c3_1 = {
        'sportsett_id': [4921, 4922, 4923, 4924, 4925, 4931, 4936, 4937, 4942, 4944],
        'team_name': ["76ers", "Lakers", "Celtics", "Heat", "Warriors", "Knicks", "Bulls", "Magic", "Spurs", "Suns"]
    }
    c3_2 = {
        'sportsett_id': [4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930],
        'team_name': ["76ers", "76ers", "Lakers", "Celtics", "Heat", "Warriors", "Knicks", "Bulls", "Magic", "Spurs"]
    }
    c3_3 = {
        'sportsett_id': [4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930],
        'team_name': ["76ers", "76ers", "76ers", "76ers", "76ers", "Lakers", "Celtics", "Heat", "Warriors", "Knicks"]
    }
    evaluate_joins("team_won_game", gt_df_3, [c3_1, c3_2, c3_3], ['sportsett_id', 'team_name'], "summaries", "teams", summaries_df, "is_winner == True")

    # ---------------------------------------------------------
    # 4. Player scored the most points
    # ---------------------------------------------------------
    gt_df_4 = pd.read_csv("datasets/nba/quality_exps/players_summaries_stats.csv")[['sportsett_id', 'player_name', 'points']]
    gt_df_4["is_top_scorer"] = gt_df_4["points"] == gt_df_4.groupby("sportsett_id")["points"].transform("max")

    c4_1 = {
        'sportsett_id': [4921, 4922, 4923, 4924, 4928, 4929, 4931, 4932, 4935, 4939],
        'player_name': ["Robert Covington", "Joel Embiid", "Kemba Walker", "Ben Simmons", "Jimmy Butler", "Danilo Gallinari", "Joe Ingles", "Trevor Ariza", "Collin Sexton", "Mike Muscala"]
    }
    c4_2 = {
        'sportsett_id': [4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930],
        'player_name': ["Joel Embiid", "Joel Embiid", "Joel Embiid", "Joel Embiid", "Joel Embiid", "Danilo Gallinari", "Joe Ingles", "Trevor Ariza", "Collin Sexton", "Mike Muscala"]
    }
    c4_3 = {
        'sportsett_id': [4922, 4922, 4922, 4922, 4922, 4925, 4925, 4925, 4925, 4925],
        'player_name': ["Joel Embiid", "Joel Embiid", "Joel Embiid", "Joel Embiid", "Joel Embiid", "Danilo Gallinari", "Joe Ingles", "Trevor Ariza", "Collin Sexton", "Mike Muscala"]
    }
    evaluate_joins("player_max_points", gt_df_4, [c4_1, c4_2, c4_3], ['sportsett_id', 'player_name'], "summaries", "players", summaries_df, "is_top_scorer == True")

    # ---------------------------------------------------------
    # 5. Player had the most assists
    # ---------------------------------------------------------
    gt_df_5 = pd.read_csv("datasets/nba/quality_exps/players_summaries_stats.csv")[['sportsett_id', 'player_name', 'assists']]
    gt_df_5["max_assists"] = gt_df_5["assists"] == gt_df_5.groupby("sportsett_id")["assists"].transform("max")

    c5_1 = {
        'sportsett_id': [4922, 4923, 4926, 4937, 4939, 4921, 4924, 4925, 4927, 4928],
        'player_name': ["J.J. Redick", "Kemba Walker", "Blake Griffin", "Thaddeus Young", "Jimmy Butler", "Robert Covington", "Joel Embiid", "Evan Fournier", "Aaron Gordon", "Markelle Fultz"]
    }
    c5_2 = {
        'sportsett_id': [4921, 4924, 4925, 4927, 4928, 4929, 4930, 4933, 4934, 4935],
        'player_name': ["Ben Simmons", "Ben Simmons", "Robert Covington", "Joel Embiid", "Evan Fournier", "Aaron Gordon", "Markelle Fultz", "Joe Harris", "Luke Kennard", "John Wall"]
    }
    c5_3 = {
        'sportsett_id': [4921, 4924, 4925, 4927, 4928, 4929, 4930, 4933, 4934, 4935],
        'player_name': ["Ben Simmons", "Ben Simmons", "Ben Simmons", "Ben Simmons", "Ben Simmons", "Robert Covington", "Joel Embiid", "Evan Fournier", "Aaron Gordon", "Markelle Fultz"]
    }
    evaluate_joins("player_max_assists", gt_df_5, [c5_1, c5_2, c5_3], ['sportsett_id', 'player_name'], "summaries", "players", summaries_df, "max_assists == True")

    # ---------------------------------------------------------
    # 6. Player had the most (total) rebounds
    # ---------------------------------------------------------
    gt_df_6 = pd.read_csv("datasets/nba/quality_exps/players_summaries_stats.csv")[['sportsett_id', 'player_name', 'total_rebounds']]
    gt_df_6["max_rebounds"] = gt_df_6["total_rebounds"] == gt_df_6.groupby("sportsett_id")["total_rebounds"].transform("max")

    c6_1 = {
        'sportsett_id': [4921, 4922, 4923, 4926, 4932, 4935, 4936, 4937, 4938, 4939],
        'player_name': ["Kemba Walker", "Aaron Gordon", "Joel Embiid", "Blake Griffin", "Mike Muscala", "Robert Covington", "Bobby Portis", "J.J. Redick", "Evan Fournier", "Markelle Fultz"]
    }
    c6_2 = {
        'sportsett_id': [4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930],
        'player_name': ["Joel Embiid", "Joel Embiid", "Joel Embiid", "Joel Embiid", "Robert Covington", "Bobby Portis", "J.J. Redick", "Evan Fournier", "Markelle Fultz", "Kemba Walker"]
    }
    c6_3 = {
        'sportsett_id': [4923, 4923, 4923, 4923, 4923, 4925, 4925, 4925, 4925, 4925],
        'player_name': ["Joel Embiid", "Joel Embiid", "Joel Embiid", "Joel Embiid", "Joel Embiid", "Robert Covington", "Bobby Portis", "J.J. Redick", "Evan Fournier", "Markelle Fultz"]
    }
    evaluate_joins("player_max_rebounds", gt_df_6, [c6_1, c6_2, c6_3], ['sportsett_id', 'player_name'], "summaries", "players", summaries_df, "max_rebounds == True")

    # ---------------------------------------------------------
    # 7. Player went to College
    # ---------------------------------------------------------
    gt_df_7 = pd.read_csv("datasets/nba/quality_exps/players_info.csv")[['player_name', 'college']].dropna(subset=['college'])

    c7_1 = {
        'college': ["Indiana", "North Carolina", "California", "Iowa State", "Stanford", "Yale", "Harvard", "Princeton", "Brown", "Cornell"],
        'player_name': ["Victor Oladipo", "Wayne Ellington", "Tyrone Wallace", "Tyrese Haliburton", "Tyrell Terry", "Zach LaVine", "Tyus Jones", "Tyler Herro", "Will Barton", "Frank Kaminsky"]
    }
    c7_2 = {
        'college': ["Duke", "Duke", "Duke", "Duke", "Yale", "Harvard", "Princeton", "Brown", "Cornell", "Dartmouth"],
        'player_name': ["Vernon Carey Jr.", "Tyus Jones", "Wendell Carter Jr.", "Frank Jackson", "Gary Trent Jr.", "Victor Oladipo", "Wayne Ellington", "Tyrone Wallace", "Tyrese Haliburton", "Tyrell Terry"]
    }
    c7_3 = {
        'college': ["Duke", "Duke", "Duke", "Duke", "Duke", "Kentucky", "Kentucky", "Kentucky", "Kentucky", "Kentucky"],
        'player_name': ["Vernon Carey Jr.", "Tyus Jones", "Wendell Carter Jr.", "Frank Jackson", "Gary Trent Jr.", "Tyrese Maxey", "Tyler Herro", "Willie Cauley-Stein", "Eric Bledsoe", "Devin Booker"]
    }
    evaluate_joins("player_college", gt_df_7, [c7_1, c7_2, c7_3], ['player_name', 'college'], "colleges", "players", summaries_df)

    # ---------------------------------------------------------
    # 8. Player was over 22 y.o. in season
    # ---------------------------------------------------------
    gt_df_8 = pd.read_csv("datasets/nba/quality_exps/all_seasons.csv")[['player_name', 'age', 'season']]

    c8_1 = {
        'season': ["2013-14", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
        'player_name': ["Shane Battier", "Ray Allen", "Chauncey Billups", "Derek Fisher", "Rashard Lewis", "Randy Livingston", "George Zidek", "Gheorghe Muresan", "Greg Dreiling", "Fred Roberts"]
    }
    c8_2 = {
        'season': ["2013-14", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
        'player_name': ["Manu Ginobili", "Paul Pierce", "Kobe Bryant", "Tim Duncan", "Kevin Garnett", "Steve Nash", "Chauncey Billups", "Randy Livingston", "George Zidek", "Gheorghe Muresan"]
    }
    c8_3 = {
        'season': ["2013-14", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
        'player_name': ["LeBron James", "Stephen Curry", "Chris Paul", "DeMar DeRozan", "Paul Pierce", "Kobe Bryant", "Tim Duncan", "Randy Livingston", "George Zidek", "Gheorghe Muresan"]
    }
    evaluate_joins("player_over_22", gt_df_8, [c8_1, c8_2, c8_3], ['player_name', 'season'], "seasons", "players", summaries_df, "age > 22")

    # ---------------------------------------------------------
    # 9. Player was born in Country
    # ---------------------------------------------------------
    gt_df_9 = pd.read_csv("datasets/nba/quality_exps/all_seasons.csv")[['player_name', 'country']]

    c9_1 = {
        'country': ["China", "Spain", "France", "Argentina", "Brazil", "Italy", "Serbia", "Greece", "Japan", "Senegal"],
        'player_name': ["Wang Zhi-zhi", "LeBron James", "Kobe Bryant", "Kevin Durant", "Stephen Curry", "James Harden", "Chris Paul", "Russell Westbrook", "Damian Lillard", "Dwyane Wade"]
    }
    c9_2 = {
        'country': ["China", "Dominican Republic", "France", "Spain", "Argentina", "Brazil", "Italy", "Serbia", "Greece", "Japan"],
        'player_name': ["Yao Ming", "Wang Zhi-zhi", "Felipe Lopez", "Mengke Bateer", "LeBron James", "Kobe Bryant", "Kevin Durant", "Stephen Curry", "James Harden", "Chris Paul"]
    }
    c9_3 = {
        'country': ["Germany", "US Virgin Islands", "Croatia", "France", "Spain", "Argentina", "Brazil", "Italy", "Serbia", "Greece"],
        'player_name': ["Dirk Nowitzki", "Tim Duncan", "Toni Kukoc", "LeBron James", "Kobe Bryant", "Kevin Durant", "Stephen Curry", "James Harden", "Chris Paul", "Dwyane Wade"]
    }
    evaluate_joins("player_country", gt_df_9, [c9_1, c9_2, c9_3], ['player_name', 'country'], "countries", "players", summaries_df)


if __name__ == "__main__":
    os.makedirs("datasets/nba", exist_ok=True)
    os.makedirs("datasets/nba/quality_exps", exist_ok=True)
    
    # process_game_summaries_all()
    # process_game_summaries()
    # process_player_info()
    process_player_summaries()
    # process_team_summaries()
    # evaluate_all_joins()
    
    print("All pre-processing tasks completed successfully!")