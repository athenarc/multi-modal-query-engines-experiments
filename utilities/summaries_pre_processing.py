import pandas as pd
import datasets
import unicodedata
import re

sporsett_dataset = datasets.load_dataset(
    "parquet", 
    data_files={
        "train": "https://huggingface.co/datasets/GEM/sportsett_basketball/resolve/refs/convert/parquet/default/train/*.parquet", 
        "validation": "https://huggingface.co/datasets/GEM/sportsett_basketball/resolve/refs/convert/parquet/default/validation/*.parquet", 
        "test": "https://huggingface.co/datasets/GEM/sportsett_basketball/resolve/refs/convert/parquet/default/test/*.parquet"
    }
)

def create_game_summaries_dataset(dataset):
    game_summaries = []

    for row in dataset.select(range(200)):
        game_id = row.get("sportsett_id")

        if "target" in row and row["target"]:
            summary = str(row["target"])
        elif "summaries" in row and isinstance(row["summaries"], list) and len(row["summaries"] > 0):
            summary = str(row["summaries"][0])

        sportsett_id_lookup = {row['sportsett_id']: row for row in dataset}

        # Home 
        home_name = row['teams']['home']['name']
        home_points = int(row['teams']['home']['line_score']['game']['PTS'])
        home_next_game_id = row['teams']['home']['next_game_id']
        if home_next_game_id in sportsett_id_lookup:
            home_next_game_row = sportsett_id_lookup[home_next_game_id]
            home_next_opponent = home_next_game_row['teams']['home']['name'] if home_next_game_row['teams']['home']['name'] != home_name else home_next_game_row['teams']['vis']['name']


        # Visitor
        vis_name = row['teams']['vis']['name']
        vis_points = int(row['teams']['vis']['line_score']['game']['PTS'])
        vis_next_game_id = row['teams']['vis']['next_game_id']
        if vis_next_game_id in sportsett_id_lookup:
            vis_next_game_row = sportsett_id_lookup[vis_next_game_id]
            vis_next_opponent = vis_next_game_row['teams']['home']['name'] if vis_next_game_row['teams']['home']['name'] != vis_name else vis_next_game_row['teams']['vis']['name']


        (winner_team, winner_points, loser_team, loser_points, win_home_or_vis, loss_home_or_vis, winner_next_opponent) = (
            (home_name, home_points, vis_name, vis_points, 'home', 'visitor', home_next_opponent)
            if home_points > vis_points
            else (vis_name, vis_points, home_name, home_points, 'visitor', 'home', vis_next_opponent)
        )

        total_points = home_points + vis_points
        margin = abs(home_points - vis_points)

        game_summaries.append({
            'sporsett_id': game_id,
            "summary": summary,
            "winner_team": winner_team,
            "winner_points": winner_points,
            "loser_team": loser_team,
            "loser_points": loser_points,
            "win_home_or_vis": win_home_or_vis,
            "loss_home_or_vis": loss_home_or_vis,
            "winner_next_opponent": winner_next_opponent,   
            "def_or_off": "defensive" if total_points < 210 else "offensive" ,
            "double_digit_margin": (10 <= margin < 100),
            "total_points_gt_180": total_points > 180
        })

    return game_summaries

summaries = create_game_summaries_dataset(sporsett_dataset['test'])
all_summaries_df = pd.DataFrame(summaries)

from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    base_url="http://localhost:5001/v1",
    api_key="EMPTY" 
)

def verification(row):
    prompt = f"""
            Task: Check if a sports summary mentions the next opponent of the winner team.

            Summary: "{row['summary']}"

            Question: Does the summary explicitly mention the next opponent of the winner team?
            Answer ONLY with 'Yes' or 'No'. Do not include any other words or punctuation.
            """
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.0,
            max_tokens=10,
        )

        answer = response.choices[0].message.content.strip().lower()
        return answer

    except Exception as e:
        print(f"Error processing row {row.name}: {e}")

tqdm.pandas(desc="Verification")
all_summaries_df['winner_opponent_mentioned'] = all_summaries_df.progress_apply(lambda row: verification(row), axis=1)

verified_summaries = all_summaries_df[all_summaries_df['winner_opponent_mentioned'] == 'yes']

verified_summaries = verified_summaries.head(100).drop(columns=['winner_opponent_mentioned']).reset_index(drop=True)

verified_summaries.to_csv("datasets/nba/game_summaries.csv", index=False)
print("Game Summaries input pre-processing completed. Table saved in datasets/nba/game_summaries.csv.")
