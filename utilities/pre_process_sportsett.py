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

def llm_verify(prompt, model="meta-llama/Llama-3.1-8B-Instruct"):
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

# Create Player Info Table
def process_player_info(input_csv="datasets/nba/all_seasons.csv", output_csv="datasets/nba/players_info.csv"):
    print("Processing Player Info...")
    stats = pd.read_csv(input_csv, index_col=0)
    stats = stats[stats['season'] == '2021-22']
    stats['team_name'] = stats['team_abbreviation'].map(NBA_TEAMS)
    
    stats = stats[~stats['draft_year'].isin(['Undrafted'])]
    stats = stats[~stats['draft_round'].isin(['Undrafted', '0'])]
    
    stats = stats.head(100)[['player_name', 'team_name', 'player_height', 'country', 'draft_year', 'draft_round', 'season']]
    
    stats['country_continent'] = stats['country'].map(COUNTRY_TO_CONTINENT)
    stats['born_in_america'] = stats['country_continent'] == 'America'
    stats['height_lt_200'] = stats['player_height'] < 200
    stats['first_round_drafted'] = stats['draft_round'] == '1'
    
    stats.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

# Create Players-Summaries Table
def process_player_summaries(output_csv="datasets/nba/players_summaries_stats.csv"):
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
                
                csv_data.append({
                    "sportsett_id": game_id, "summary": summary, "player_name": player_name,
                    "team": team_name, "points": int(pts), "assists": int(ast), "total_rebounds": int(reb),
                    "points_gt_20": int(pts) > 20, "is_mentioned": player_name in summary
                })

    df = pd.DataFrame(csv_data)
    df.to_csv("datasets/nba/players_summaries_all.csv", index=False)
    
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

    verified_summaries = df[(df['points'] != 0) & (df['assists'] != 0) & (df['total_rebounds'] != 0)]
    verified_summaries.head(100).drop(columns=['points_verified', 'assists_verified', 'rebounds_verified']).reset_index(drop=True).to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

# Create Game Summaries Table
def process_game_summaries(output_csv="datasets/nba/game_summaries.csv"):
    print("Processing Game Summaries...")
    game_summaries = []
    
    sportsett_id_lookup = {row['sportsett_id']: row for row in test_data}

    for row in test_data.select(range(200)):
        game_id = row.get("sportsett_id")
        summary = get_summary(row)

        def get_next_opponent(team_data):
            next_game_id = team_data['next_game_id']
            if next_game_id in sportsett_id_lookup:
                next_row = sportsett_id_lookup[next_game_id]
                return next_row['teams']['home']['name'] if next_row['teams']['home']['name'] != team_data['name'] else next_row['teams']['vis']['name']
            return None

        home = row['teams']['home']
        vis = row['teams']['vis']
        
        home_pts = int(home['line_score']['game']['PTS'])
        vis_pts = int(vis['line_score']['game']['PTS'])

        if home_pts > vis_pts:
            winner, loser = (home['name'], home_pts, 'home', get_next_opponent(home)), (vis['name'], vis_pts, 'visitor')
        else:
            winner, loser = (vis['name'], vis_pts, 'visitor', get_next_opponent(vis)), (home['name'], home_pts, 'home')

        total_points = home_pts + vis_pts
        
        game_summaries.append({
            'sporsett_id': game_id, "summary": summary,
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

# Create Teams-Summaries Table
def process_team_summaries(output_csv="datasets/nba/teams_summaries.csv"):
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


if __name__ == "__main__":
    process_player_info()
    process_player_summaries()
    process_game_summaries()
    process_team_summaries()
    print("All tasks completed!")