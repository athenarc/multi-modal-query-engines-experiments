import os
import unicodedata
import pandas as pd
import datasets
from openai import OpenAI
from tqdm import tqdm

tqdm.pandas(desc="Verification")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="EMPTY")

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
train_data = sportsett_dataset['train']
data = datasets.concatenate_datasets([test_data, train_data])

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

def llm_verify(prompt, model="llama3:8b-instruct-q8_0"):
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
def process_player_info(input_csv="datasets/nba/all_seasons.csv", output_csv="datasets/nba/scalability_exps/players_info.csv"):
    print("Processing Player Info...")
    stats = pd.read_csv(input_csv, index_col=0)
        
    stats = stats.head(4000)[['player_name', 'team_name', 'player_height', 'country', 'draft_year', 'draft_round', 'college', 'age', 'season']]

    stats.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

def process_player_summaries(output_csv="datasets/nba/scalability_exps/players_summaries_stats.csv"):
    print("Processing Player Summaries...")
    csv_data = []
    
    for row in data:
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

    verified_summaries.head(4000)[['sportsett_id', 'player_name', 'summary']].to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

def process_game_summaries(output_csv="datasets/nba/scalability_exps/game_summaries.csv"):
    print("Processing Game Summaries...")
    game_summaries = []
    
    for row in data:
        game_id = row.get("sportsett_id")
        summary = get_summary(row)

        home = row['teams']['home']
        vis = row['teams']['vis']
        
        home_pts = int(home['line_score']['game']['PTS'])
        vis_pts = int(vis['line_score']['game']['PTS'])

        if home_pts > vis_pts:
            winner, loser = (home['name'], home_pts, 'home', vis['next_game']['opponent_name']), (vis['name'], vis_pts, 'visitor')
        else:
            winner, loser = (vis['name'], vis_pts, 'visitor', home['next_game']['opponent_name']), (home['name'], home_pts, 'home')

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
    verified[['sportsett_id', 'summary']].to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

def process_team_summaries(output_csv="datasets/nba/scalability/teams_summaries.csv", output_csv_all="datasets/nba/scalability/teams_summaries_all.csv"):
    print("Processing Team Summaries...")
    csv_data = []

    for row in data:
        game_id = row.get("sportsett_id")
        summary = get_summary(row)

        home_name = row['teams']['home']['name']
        vis_name = row['teams']['vis']['name']

        for team in [home_name, vis_name]:
            csv_data.append({
                "sportsett_id": game_id,
                "team": team,
                "summary": summary,
            })

    pd.DataFrame(csv_data).to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Saved to {output_csv}")

    csv_data = []

    for row in data:
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

if __name__ == "__main__":
    os.makedirs("datasets/nba", exist_ok=True)
    os.makedirs("datasets/nba/scalability_exps", exist_ok=True)
    
    process_player_info()
    process_player_summaries()
    process_game_summaries()
    process_team_summaries()
    # evaluate_all_joins()
    
    print("All pre-processing tasks completed successfully!")