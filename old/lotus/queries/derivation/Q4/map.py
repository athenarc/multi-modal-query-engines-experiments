from datetime import datetime
import pandas as pd
import lotus
from lotus.models import LM
from time import time
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q4_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="SQE_experiments",
        name=run_name,
        group="Derivation",
    )

if args.provider == 'ollama':
    lm = LM(args.provider + '/' + args.model, caching=False)
elif args.provider == 'vllm':
    lm = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000, caching=False)

lotus.settings.configure(lm=lm)

df_reports = pd.read_csv('datasets/rotowire/reports_table.csv').rename(columns={'Game_ID': 'Game ID'})
df_player_names = pd.read_csv('datasets/rotowire/player_labels.csv')[['Player Name', 'Game ID']].head(args.size)
df_init = pd.merge(df_player_names, df_reports, on='Game ID')

print(df_init)

elapsed_times = []

# Extract assists
print("Extracting Assists")
examples = {
    "Player Name": ["Tim Hardaway Jr"],
    "Report": ["The Milwaukee Bucks (18 - 17) defeated the New York Knicks (5 - 31) 95 - 82 on Sunday at Madison Square Garden in New York. The Bucks were able to have a great night defensively, giving themselves the scoring advantage in all four quarters. The Bucks showed superior shooting, going 46 percent from the field, while the Knicks went only 41 percent from the floor. The Bucks also out - rebounded the Knicks 48 - 36, giving them in an even further advantage which helped them secure the 13 - point victory on the road. Brandon Knight led the Bucks again in this one. He went 6 - for - 14 from the field and 1 - for - 3 from beyond the arc to score 17 points, while also handing out five assists. He's now averaging 21 points per game over his last three games, as he's consistently been the offensive leader for this team. Zaza Pachulia also had a strong showing, finishing with 16 points (6 - 12 FG, 4 - 4 FT) and a team - high of 14 rebounds. It marked his second double - double in a row and fourth on the season, as the inexperienced centers on the Knicks' roster were n't able to limit him. Notching a double - double of his own, Giannis Antetokounmpo recorded 16 points (6 - 9 FG, 1 - 1 3Pt, 3 - 6 FT) and 12 rebounds. The 12 rebounds matched a season - high, while it was his second double - double of the season. Coming off the bench for a big night was Kendall Marshall. He went 6 - for - 8 from the field and 3 - for - 3 from the free throw line to score 15 points in 20 minutes. The Knicks really struggled to score without Carmelo Anthony and Amare Stoudemire. Tim Hardaway Jr led the team as the starting shooting guard, going 6 - for - 13 from the field and 3 - for - 5 from the three - point line to score 17 points, while also adding four assists. He's now scored 17 or more points in three out of his last four games, as he has put it on himself to pick up the slack with other key players sitting out. J.R. Smith also put together a solid outing as a starter. He finished with 15 points and seven rebounds in 37 minutes. Like Haradaway Jr, he's also benefitted from other guys sitting out, and has now combined for 37 points over his last two games. While he did n't have his best night defensively, Cole Aldrich scored 12 points (6 - 10 FG) and grabbed seven rebounds in 19 minutes. The only other Knick to reach double figures in points was Jason Smith, who came off the bench for 10 points (3 - 11 FG, 4 - 4 FT). The Bucks' next game will be at home against the Phoenix Suns on Tuesday, while the Knicks will travel to Memphis to play the Grizzlies on Monday."],
    "Answer": [4]
}
user_instruction = "What is the number of assists for {Player Name} in the game {Report} if they are mentioned. Return only the number (only an integer) of assists or -1 if there are no mentions for assists for {Player Name} (without explanation)."
start = time()
df_assists = df_init.sem_map(user_instruction)
end = time()
elapsed_times.append(end-start)
print("Assists Extraction, Elapsed Time: ", end-start, '\n')
df = df_assists.rename(columns={"_map": "assists"})
print(df)

# Extract total rebounds
print("Extracting Total Rebounds")
examples = {
    "Player Name": ["Zaza Pachulia"],
    "Report": ["The Milwaukee Bucks (18 - 17) defeated the New York Knicks (5 - 31) 95 - 82 on Sunday at Madison Square Garden in New York. The Bucks were able to have a great night defensively, giving themselves the scoring advantage in all four quarters. The Bucks showed superior shooting, going 46 percent from the field, while the Knicks went only 41 percent from the floor. The Bucks also out - rebounded the Knicks 48 - 36, giving them in an even further advantage which helped them secure the 13 - point victory on the road. Brandon Knight led the Bucks again in this one. He went 6 - for - 14 from the field and 1 - for - 3 from beyond the arc to score 17 points, while also handing out five assists. He's now averaging 21 points per game over his last three games, as he's consistently been the offensive leader for this team. Zaza Pachulia also had a strong showing, finishing with 16 points (6 - 12 FG, 4 - 4 FT) and a team - high of 14 rebounds. It marked his second double - double in a row and fourth on the season, as the inexperienced centers on the Knicks' roster were n't able to limit him. Notching a double - double of his own, Giannis Antetokounmpo recorded 16 points (6 - 9 FG, 1 - 1 3Pt, 3 - 6 FT) and 12 rebounds. The 12 rebounds matched a season - high, while it was his second double - double of the season. Coming off the bench for a big night was Kendall Marshall. He went 6 - for - 8 from the field and 3 - for - 3 from the free throw line to score 15 points in 20 minutes. The Knicks really struggled to score without Carmelo Anthony and Amare Stoudemire. Tim Hardaway Jr led the team as the starting shooting guard, going 6 - for - 13 from the field and 3 - for - 5 from the three - point line to score 17 points, while also adding four assists. He's now scored 17 or more points in three out of his last four games, as he has put it on himself to pick up the slack with other key players sitting out. J.R. Smith also put together a solid outing as a starter. He finished with 15 points and seven rebounds in 37 minutes. Like Haradaway Jr, he's also benefitted from other guys sitting out, and has now combined for 37 points over his last two games. While he did n't have his best night defensively, Cole Aldrich scored 12 points (6 - 10 FG) and grabbed seven rebounds in 19 minutes. The only other Knick to reach double figures in points was Jason Smith, who came off the bench for 10 points (3 - 11 FG, 4 - 4 FT). The Bucks' next game will be at home against the Phoenix Suns on Tuesday, while the Knicks will travel to Memphis to play the Grizzlies on Monday."],
    "Answer": [14]
}
user_instruction = "What is the number of total rebounds for {Player Name} in the game {Report} if they are mentioned. Return only the number (only an integer) of total rebounds or -1 if there are no mentions for total rebounds for {Player Name} (without explanation)."
start = time()
df_rebounds = df.sem_map(user_instruction)
end = time()
elapsed_times.append(end-start)
print("Total Rebounds Extraction, Elapsed Time: ", end-start, '\n')
df = df_rebounds.rename(columns={"_map": "total_rebounds"})

print(df)

# Extract blocks
print("Extracting Blocks")
examples = {
    "Player Name": ["Jonas Valnciunas"],
    "Report": ["The Toronto Raptors (29 - 15) defeated the Detroit Pistons (17 - 28) 114 - 110 on Sunday at the Air Canada Center in Toronto. Despite being out - scored 31 - 25 in the final quarter, the Raptors were able to hold of the Pistons' late comeback attempt and secure the four - point victory in front of their home crowd. While the game may have been close, the Raptors shot the ball much better than the Pistons, going 53 percent from the field compared to just 46 percent from the field for the Pistons. The Raptors also forced them into 14 turnovers, while only committing eight of their own, which may have made a big difference in this one. After combining for only 14 points over his last three games, DeMar DeRozan returned to form Sunday, finishing with 25 points (8 - 14 FG, 1 - 2 3Pt, 8 - 10 FT), six rebounds and four assists. It was good to see him turn things around, as the Raptors really needed him to play well after losing six of their last ten games. Jonas Valnciunas was another big factor in the win. He went 9 - for - 15 from the field and 2 - for - 2 from the free throw line to score 20 points, while adding 11 rebounds and three blocked shots as well. He's now recorded a double - double in three out of his last four games, while also notching three blocks in two consecutive outings. Amir Johnson had a solid showing as well, finishing with an efficient 17 points (7 - 9 FG, 3 - 4 FT) and two rebounds. He only played 17 minutes in Friday's win over the Sixers, but he was back to a normal amount of minutes Sunday, playing a full game's worth of 28. Both Greivis Vasquez and Louis Williams reached double figures in points as well, with 13 and 12 points respectively. With Brandon Jennings going down for the year, D.J. Augustin stepped up in a big way, going 12 - for - 20 from the field and 5 - for - 9 from the three - point line to score a game - high of 35 points, while adding eight assists as well. It was the most shots he's taken all season, resulting in a new season - high in points. He'll run as the starting point guard moving forward. Greg Monroe had another strong stat line Sunday, recording 21 points (9 - 17 FG, 3 - 5 FT) and 16 rebounds. He's now posted a double - double in four out of his last five games. Andre Drummond nearly notched a double - double of his own, but came up just shy with 14 points (7 - 11 FG, 0 - 1 FT) and eight rebounds. He had a really tough matchup with Valanciunas, so he did n't have his normal eye - popping amount of rebounds that he's come accustomed to. The only other Piston to reach double figures in points was Kentavious Caldwell-Pope who added 16 points (6 - 17 FG, 3 - 9 3Pt, 1 - 2 FT), three rebounds and two steals. The Raptors' next game will be on the road against the Indiana Pacers on Tuesday, while the Pistons will be at home against the Cleveland Cavaliers on Tuesday."],
    "Answer": [3]
}
user_instruction = "What is the number of blocks for {Player Name} in the game {Report} if they are mentioned. Return only the number (only an integer) of blocks or -1 if there are no mentions for blocks for {Player Name} (without explanation)."
start = time()
df_blocks = df.sem_map(user_instruction)
end = time()
elapsed_times.append(end-start)
print("Blocks Extraction, Elapsed Time: ", end-start, '\n')
df = df_blocks.rename(columns={"_map": "blocks"})

print(df)

# Extract steals
print("Extracting Steals")
examples = {
    "Player Name": ["Kentavious Caldwell-Pope"],
    "Report": ["The Toronto Raptors (29 - 15) defeated the Detroit Pistons (17 - 28) 114 - 110 on Sunday at the Air Canada Center in Toronto. Despite being out - scored 31 - 25 in the final quarter, the Raptors were able to hold of the Pistons' late comeback attempt and secure the four - point victory in front of their home crowd. While the game may have been close, the Raptors shot the ball much better than the Pistons, going 53 percent from the field compared to just 46 percent from the field for the Pistons. The Raptors also forced them into 14 turnovers, while only committing eight of their own, which may have made a big difference in this one. After combining for only 14 points over his last three games, DeMar DeRozan returned to form Sunday, finishing with 25 points (8 - 14 FG, 1 - 2 3Pt, 8 - 10 FT), six rebounds and four assists. It was good to see him turn things around, as the Raptors really needed him to play well after losing six of their last ten games. Jonas Valnciunas was another big factor in the win. He went 9 - for - 15 from the field and 2 - for - 2 from the free throw line to score 20 points, while adding 11 rebounds and three blocked shots as well. He's now recorded a double - double in three out of his last four games, while also notching three blocks in two consecutive outings. Amir Johnson had a solid showing as well, finishing with an efficient 17 points (7 - 9 FG, 3 - 4 FT) and two rebounds. He only played 17 minutes in Friday's win over the Sixers, but he was back to a normal amount of minutes Sunday, playing a full game's worth of 28. Both Greivis Vasquez and Louis Williams reached double figures in points as well, with 13 and 12 points respectively. With Brandon Jennings going down for the year, D.J. Augustin stepped up in a big way, going 12 - for - 20 from the field and 5 - for - 9 from the three - point line to score a game - high of 35 points, while adding eight assists as well. It was the most shots he's taken all season, resulting in a new season - high in points. He'll run as the starting point guard moving forward. Greg Monroe had another strong stat line Sunday, recording 21 points (9 - 17 FG, 3 - 5 FT) and 16 rebounds. He's now posted a double - double in four out of his last five games. Andre Drummond nearly notched a double - double of his own, but came up just shy with 14 points (7 - 11 FG, 0 - 1 FT) and eight rebounds. He had a really tough matchup with Valanciunas, so he did n't have his normal eye - popping amount of rebounds that he's come accustomed to. The only other Piston to reach double figures in points was Kentavious Caldwell-Pope who added 16 points (6 - 17 FG, 3 - 9 3Pt, 1 - 2 FT), three rebounds and two steals. The Raptors' next game will be on the road against the Indiana Pacers on Tuesday, while the Pistons will be at home against the Cleveland Cavaliers on Tuesday."],
    "Answer": [2]
}
user_instruction = "What is the number of steals for {Player Name} in the game {Report} if they are mentioned. Return only the number (only an integer) of steals or -1 if there are no mentions for steals for {Player Name} (without explanation)."
start = time()
df_steals = df.sem_map(user_instruction)
end = time()
elapsed_times.append(end-start)
print("Steals Extraction, Elapsed Time: ", end-start, '\n')
df = df_steals.rename(columns={"_map": "steals"})

print(df)

exec_time = sum(elapsed_times)

if args.provider == 'ollama':
    output_file = f"evaluation/derivation/Q4/results/lotus_Q4_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider =='vllm':
    output_file = f"evaluation/derivation/Q4/results/lotus_Q4_map_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
df.to_csv(output_file)

num_extraction_attributes = 4  # assists, total_rebounds, blocks, steals

total_LLM_calls = df_init.shape[0] * num_extraction_attributes

with open('statistics/derivation/Q4.log', 'a') as file:
    file.write(f"System: Lotus (sem_map)\n")
    file.write(f"Timestamp: {datetime.now().isoformat()}\n")
    file.write(f"Model: {args.model}\n")
    file.write("Execution Time: " + str(exec_time) + "\n")
    file.write("Total LLM calls: " + str(total_LLM_calls) + "\n")

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time,
        "total_LLM_calls": total_LLM_calls
    })
    wandb.finish()