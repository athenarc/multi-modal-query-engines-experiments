import pandas as pd

### Ground Truth Data for Aggregation Queries on NBA dataset
print("NBA Dataset Aggregation Ground Truth Data")

game_4922 = pd.DataFrame({
    "player_name": [
        "Joel Embiid",
        "J.J. Redick",
        "Dario Saric",
        "Robert Covington",
        "Evan Fournier",
        "Nikola Vucevic",
        "Aaron Gordon"
    ],
    "points": [32, 31, 13, 12, 31, 27, 20],
    "assists": [3, 4, 5, 1, 3, 12, 3],
    "total_rebounds": [10, 4, 9, 4, 4, 13, 12],
})

game_4923 = pd.DataFrame({
    "player_name": [
        "Kemba Walker",
        "Miles Bridges",
        "Cody Zeller",
        "Jeremy Lamb",
        "Joel Embiid",
        "Robert Covington",
        "J.J. Redick",
        "Ben Simmons",
        "Dario Saric",
        "Markelle Fultz"
    ],
    "points": [37, 14, 12, 12, 27, 18, 15, 14, 11, 10],
    "assists": [6, 0, 0, 1, 2, 0, 0, 3, 1, 4],
    "total_rebounds": [6, 2, 6, 3, 14, 10, 0, 12, 7, 4]
})

game_4925 = pd.DataFrame({
    "player_name": [
        "Joel Embiid",
        "J.J. Redick",
        "Ben Simmons",
        "Lou Williams",
        "Danilo Gallinari",
        "Tobias Harris"
    ],
    "points": [41, 18, 14, 26, 25, 24],
    "assists": [1, 0, 11, 0, 1, 3],
    "total_rebounds": [13, 0, 3, 0, 8, 8]
})

game_4926 = pd.DataFrame({
    "player_name": [
        "Blake Griffin",
        "Andre Drummond",
        "Langston Galloway",
        "Joel Embiid",
        "Ben Simmons",
        "J.J. Redick"
    ],
    "points": [38, 8, 13, 39, 9, 16],
    "assists": [6, 0, 0, 0, 5, 4],
    "total_rebounds": [13, 9, 0, 17, 5, 0]
})

game_4927 = pd.DataFrame({
    "player_name": [
        "Kemba Walker",
        "Jeremy Lamb",
        "Dwayne Bacon",
        "Willy Hernangomez",
        "Cody Zeller",
        "Michael Kidd-Gilchrist",
        "Joel Embiid",
        "Ben Simmons",
        "Dario Saric",
        "J.J. Redick",
        "Markelle Fultz",
        "Robert Covington"
    ],
    "points": [30, 17, 15, 14, 14, 12, 42, 22, 18, 17, 7, 7],
    "assists": [9, 1, 1, 0, 0, 0, 4, 13, 2, 4, 0, 0],
    "total_rebounds": [7, 3, 1, 4, 4, 12, 18, 8, 9, 2, 0, 0]
})

game_4928 = pd.DataFrame({
    "player_name": [
        "Donovan Mitchell",
        "Joe Ingles",
        "Ricky Rubio",
        "Derrick Favors",
        "Rudy Gobert",
        "Jimmy Butler",
        "Joel Embiid",
        "J.J. Redick",
        "Ben Simmons",
        "Amir Johnson"
    ],
    "points": [31, 14, 13, 13, 12, 28, 23, 16, 10, 11],
    "assists": [0, 2, 0, 0, 0, 7, 2, 0, 8, 0],
    "total_rebounds": [2, 2, 0, 0, 10, 3, 7, 0, 8, 0]
})


game_4929 = pd.DataFrame({
    "player_name": [
        "Devin Booker",
        "T.J. Warren",
        "Deandre Ayton",
        "Mikal Bridges",
        "Trevor Ariza",
        "Richaun Holmes",
        "Joel Embiid",
        "Ben Simmons",
        "Mike Muscala",
        "J.J. Redick",
        "Jimmy Butler"
    ],
    "points": [37, 21, 17, 13, 10, 10, 33, 19, 19, 17, 16],
    "assists": [8, 4, 1, 0, 5, 0, 1, 9, 0, 3, 1],
    "total_rebounds": [3, 4, 9, 0, 7, 0, 17, 11, 0, 4, 3]
})

game_4930 = pd.DataFrame({
    "player_name": [
        "Joel Embiid",
        "Ben Simmons",
        "Jimmy Butler",
        "J.J. Redick",
        "Jrue Holiday",
        "E'Twaun Moore",
        "Julius Randle"
    ],
    "points": [31, 22, 13, 13, 30, 30, 22],
    "assists": [2, 7, 0, 0, 0, 0, 3],
    "total_rebounds": [19, 8, 0, 0, 0, 0, 10]
})

game_4931 = pd.DataFrame({
    "player_name": [
        "Rodney Hood",
        "Collin Sexton",
        "Tristan Thompson",
        "Joel Embiid",
        "J.J. Redick",
        "Ben Simmons",
        "Jimmy Butler"
    ],
    "points": [25, 23, 18, 24, 23, 22, 22],
    "assists": [1, 3, 0, 3, 0, 0, 0],
    "total_rebounds": [4, 5, 13, 12, 0, 0, 0]
})

game_4932 = pd.DataFrame({
    "player_name": [
        "Enes Kanter",
        "Mario Hezonja",
        "Damyean Dotson",
        "Tim Hardaway Jr.",
        "Joel Embiid",
        "J.J. Redick",
        "Ben Simmons"
    ],
    "points": [17, 17, 16, 5, 26, 24, 14],
    "assists": [0, 0, 0, 0, 7, 2, 7],
    "total_rebounds": [6, 5, 4, 0, 14, 2, 8]
})

game_4933 = pd.DataFrame({
    "player_name": [
        "Bradley Beal",
        "Austin Rivers",
        "Thomas Bryant",
        "John Wall",
        "Joel Embiid",
        "T.J. McConnell",
        "J.J. Redick",
        "Ben Simmons",
        "Furkan Korkmaz",
        "Landry Shamet",
        "Mike Muscala",
        "Jimmy Butler"
    ],
    "points": [19, 15, 12, 11, 16, 15, 14, 13, 13, 12, 12, 11],
    "assists": [0, 1, 0, 7, 0, 0, 0, 10, 5, 2, 0, 4],
    "total_rebounds": [0, 2, 7, 3, 15, 0, 0, 8, 2, 4, 10, 7]
})

game_4934 = pd.DataFrame({
    "player_name": [
        "J.J. Redick",
        "Jimmy Butler",
        "Ben Simmons",
        "Joel Embiid",
        "Mike Conley",
        "Jaren Jackson Jr.",
        "JaMychal Green"
    ],
    "points": [24, 21, 19, 15, 21, 17, 14],
    "assists": [0, 2, 6, 3, 5, 0, 0],
    "total_rebounds": [0, 3, 12, 14, 2, 3, 7]
})

game_4935 = pd.DataFrame({
    "player_name": [
        "Joel Embiid",
        "Ben Simmons",
        "T.J. McConnell",
        "Luke Kennard",
        "Andre Drummond"
    ],
    "points": [24, 18, 14, 28, 21],
    "assists": [3, 7, 6, 3, 2],
    "total_rebounds": [8, 10, 4, 8, 17]
})

game_4937 = pd.DataFrame({
    "player_name": [
        "Thaddeus Young",
        "Bojan Bogdanovic",
        "Victor Oladipo",
        "Domantas Sabonis",
        "Cory Joseph",
        "Darren Collison",
        "Joel Embiid",
        "J.J. Redick",
        "Ben Simmons",
        "Furkan Korkmaz"
    ],
    "points": [26, 18, 14, 14, 14, 8, 40, 22, 18, 6],
    "assists": [5, 2, 9, 0, 0, 10, 3, 0, 4, 0],
    "total_rebounds": [10, 5, 0, 16, 0, 0, 21, 0, 9, 2]
})

game_4939 = pd.DataFrame({
    "player_name": [
        "Pascal Siakam",
        "Kyle Lowry",
        "Norman Powell",
        "Joel Embiid",
        "Ben Simmons",
        "J.J. Redick",
        "Jimmy Butler"
    ],
    "points": [26, 20, 13, 27, 26, 22, 12],
    "assists": [2, 5, 0, 0, 8, 5, 7],
    "total_rebounds": [6, 6, 4, 11, 12, 3, 7]
})

game_4940 = pd.DataFrame({
    "player_name": [
        "Wesley Matthews",
        "Luka Doncic",
        "Jalen Brunson",
        "Joel Embiid",
        "J.J. Redick",
        "Ben Simmons",
        "Jonah Bolden"
    ],
    "points": [18, 14, 13, 25, 20, 20, 11],
    "assists": [0, 4, 8, 5, 2, 11, 3],
    "total_rebounds": [2, 8, 11, 12, 2, 14, 9]
})

game_4941 = pd.DataFrame({
    "player_name": [
        "Bradley Beal",
        "Jeff Green",
        "Sam Dekker",
        "Otto Porter, Jr.",
        "Ian Mahinmi",
        "Chasson Randle",
        "Landry Shamet",
        "Joel Embiid",
        "Jimmy Butler",
        "Ben Simmons",
        "Furkan Korkmaz"
    ],
    "points": [28, 15, 14, 11, 10, 10, 29, 20, 20, 17, 16],
    "assists": [2, 1, 4, 0, 5, 0, 1, 4, 3, 9, 0],
    "total_rebounds": [3, 1, 3, 0, 7, 0, 3, 10, 5, 5, 0]
})

game_4942 = pd.DataFrame({
    "player_name": [
        "Kevin Huerter",
        "John Collins",
        "Dewayne Dedmon",
        "Trae Young",
        "De'Andre Bembry",
        "Alex Len",
        "Jeremy Lin",
        "Jimmy Butler",
        "Ben Simmons",
        "J.J. Redick",
        "Mike Muscala",
        "T.J. McConnell"
    ],
    "points": [29, 25, 19, 18, 14, 9, 9, 30, 23, 20, 16, 16],
    "assists": [3, 2, 7, 5, 5, 0, 0, 5, 15, 2, 0, 6],
    "total_rebounds": [3, 9, 8, 4, 6, 0, 0, 4, 10, 1, 3, 3]
})

game_4943 = pd.DataFrame({
    "player_name": [
        "Derrick Rose",
        "Karl-Anthony Towns",
        "Andrew Wiggins",
        "Jeff Teague",
        "Dario Saric",
        "Luol Deng",
        "Taj Gibson",
        "Joel Embiid",
        "Ben Simmons",
        "Jimmy Butler",
        "J.J. Redick",
        "Wilson Chandler",
        "Jonah Bolden",
        "Landry Shamet"
    ],
    "points": [15, 13, 12, 11, 11, 11, 10, 31, 20, 19, 15, 14, 14, 12],
    "assists": [4, 2, 0, 0, 0, 0, 1, 3, 9, 4, 3, 4, 0, 4],
    "total_rebounds": [2, 3, 4, 0, 0, 0, 5, 13, 11, 3, 0, 5, 0, 3]
})

game_4944 = pd.DataFrame({
    "player_name": [
        "Paul George",
        "Russell Westbrook",
        "Steven Adams",
        "Dennis Schroder",
        "Joel Embiid",
        "Ben Simmons",
        "J.J. Redick",
        "Jimmy Butler"
    ],
    "points": [31, 21, 16, 21, 31, 20, 22, 18],
    "assists": [5, 6, 3, 2, 6, 9, 0, 4],
    "total_rebounds": [6, 10, 9, 4, 8, 15, 2, 3]
})

game_4945 = pd.DataFrame({
    "player_name": [
        "Joel Embiid",
        "Landry Shamet",
        "J.J. Redick",
        "James Harden",
        "Gerald Green",
        "Kenneth Faried"
    ],
    "points": [32, 18, 16, 37, 18, 13],
    "assists": [2, 0, 1, 3, 2, 0],
    "total_rebounds": [14, 1, 2, 6, 2, 6]
})

game_4946 = pd.DataFrame({
    "player_name": [
        "DeMar DeRozan",
        "Patty Mills",
        "Rudy Gay",
        "Marco Belinelli",
        "Derrick White",
        "Bryn Forbes",
        "LaMarcus Aldridge",
        "Joel Embiid",
        "Ben Simmons",
        "J.J. Redick",
        "Landry Shamet",
        "T.J. McConnell"
    ],
    "points": [26, 17, 17, 16, 15, 14, 13, 33, 21, 19, 14, 10],
    "assists": [3, 0, 3, 1, 5, 2, 6, 3, 15, 3, 2, 8],
    "total_rebounds": [9, 0, 4, 2, 4, 6, 5, 19, 10, 3, 0, 2]
})


game_4947 = pd.DataFrame({
    "player_name": [
        "Kawhi Leonard",
        "Kyle Lowry",
        "Pascal Siakam",
        "Serge Ibaka",
        "Joel Embiid",
        "Ben Simmons",
        "Jimmy Butler",
        "Furkan Korkmaz"
    ],
    "points": [24, 20, 16, 20, 37, 20, 18, 11],
    "assists": [3, 6, 2, 3, 2, 6, 5, 0],
    "total_rebounds": [7, 0, 6, 10, 13, 7, 0, 0]
})

game_4948 = pd.DataFrame({
    "player_name": [
        "Tobias Harris",
        "J.J. Redick",
        "Jimmy Butler",
        "Nikola Jokic",
        "Jamal Murray",
        "Will Barton"
    ],
    "points": [14, 34, 22, 27, 23, 14],
    "assists": [3, 3, 5, 10, 6, 7],
    "total_rebounds": [8, 3, 7, 10, 5, 8]
})

game_4949 = pd.DataFrame({
    "player_name": [
        "Kyle Kuzma",
        "JaVale McGee",
        "Brandon Ingram",
        "LeBron James",
        "Reggie Bullock",
        "Joel Embiid",
        "Tobias Harris",
        "J.J. Redick",
        "Jimmy Butler",
        "T.J. McConnell",
        "Boban Marjanovic",
        "Ben Simmons"
    ],
    "points": [39, 21, 19, 18, 2, 37, 22, 21, 15, 13, 10, 8],
    "assists": [1, 1, 4, 9, 0, 3, 6, 5, 3, 0, 0, 7],
    "total_rebounds": [3, 13, 4, 10, 0, 14, 6, 2, 4, 0, 0, 3]
})

game_4950 = pd.DataFrame({
    "player_name": [
        "Gordon Hayward",
        "Al Horford",
        "Jayson Tatum",
        "Marcus Morris",
        "Joel Embiid",
        "Jimmy Butler",
        "J.J. Redick",
        "Ben Simmons"
    ],
    "points": [26, 23, 20, 17, 23, 22, 16, 16],
    "assists": [3, 5, 0, 2, 3, 0, 2, 5],
    "total_rebounds": [4, 8, 10, 8, 14, 9, 3, 5]
})

game_4951 = pd.DataFrame({
    "player_name": [
        "Boban Marjanovic",
        "Dwyane Wade",
        "Dion Waiters",
        "Kelly Olynyk",
        "Josh Richardson",
        "Justise Winslow",
        "Tobias Harris",
        "Ben Simmons",
        "Jimmy Butler",
        "J.J. Redick"
    ],
    "points": [19, 19, 18, 15, 13, 11, 23, 21, 18, 13],
    "assists": [2, 6, 1, 3, 5, 5, 1, 4, 6, 3],
    "total_rebounds": [12, 4, 5, 6, 2, 7, 11, 7, 6, 6]
})

games = pd.concat([
    game_4922, game_4923, game_4925, game_4926, game_4927, game_4928, game_4929, game_4930, game_4931, game_4932, game_4933, game_4934, game_4935, game_4937, game_4939, game_4940, game_4941, game_4942, game_4943, game_4944, game_4945, game_4946, game_4947, game_4948, game_4949, game_4950, game_4951
], ignore_index=True)

triple_double_mask = (games['points'] >= 10) & (games['assists'] >= 10) & (games['total_rebounds'] >= 10)
triple_doubles = games[triple_double_mask]

# Count how many triple-doubles each player has
td_counts = triple_doubles['player_name'].value_counts()

if not td_counts.empty:
    most_td_player = td_counts.idxmax()
    max_td_count = td_counts.max()
    print(f"Player with most triple-doubles: {most_td_player} ({max_td_count} triple-doubles)")
else:
    print("No players recorded a triple-double in this dataset.")

by_player = games.groupby('player_name')[['points', 'assists']].sum()

most_points_player = by_player['points'].idxmax()
max_points = by_player['points'].max()

most_assists_player = by_player['assists'].idxmax()
max_assists = by_player['assists'].max()

print(f"Player with most points: {most_points_player} ({max_points} points)")
print(f"Player with most assists: {most_assists_player} ({max_assists} assists)")

### Ground Truth Data for Aggregation Queries on Rotten Tomatoes dataset
print("\n\nRotten Tomatoes Dataset Aggregation Ground Truth Data")

# Query 64
movies = pd.read_csv("datasets/rotten_tomatoes/quality_exps/movies_derivation_curated.csv")[['movie_title', 'movie_info']].head(27)

movies['mentioned_actors'] = movies['movie_info'].str.findall(r'\((.*?)\)').apply(lambda groups:[name.strip() for g in groups for name in g.split(",")])
movies = movies.explode('mentioned_actors').dropna(subset=['mentioned_actors'])

duplicate_counts = movies["mentioned_actors"].value_counts()
duplicate_counts = duplicate_counts[duplicate_counts > 1]

print("The maximum number of times an actor appears in movies: ", duplicate_counts.max())

# Query 65
reviews = pd.read_csv("datasets/rotten_tomatoes/quality_exps/reviews_derivation_curated.csv")[['rotten_tomatoes_link', 'review_type']].head(27)
print("The most common review type is: ", reviews['review_type'].value_counts().idxmax())

# Query 66
movies = pd.read_csv("datasets/rotten_tomatoes/quality_exps/movies_derivation_curated.csv")[['movie_title', 'movie_info', 'genre']].head(27)
print("The most common genre is: ", movies['genre'].value_counts().idxmax())

