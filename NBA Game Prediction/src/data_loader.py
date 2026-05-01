import pandas as pd
import os
from nba_api.stats.endpoints import leaguegamefinder

def load_nba_games(save_path="data/raw/games_raw.csv"):
    """
    Fetch NBA games with basic error handling and save locally.
    """
    try:
        print("Fetching NBA data...")
        gamefinder = leaguegamefinder.LeagueGameFinder()
        games = gamefinder.get_data_frames()[0]
    except Exception as e:
        print("Error fetching NBA data:", e)
        raise

    # Select relevant columns
    games = games[['GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION',
                   'PTS', 'REB', 'AST', 'TOV', 'FG_PCT', 'WL']]

    # Convert types
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games['WL'] = games['WL'].map({'W': 1, 'L': 0})

    # Save raw data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    games.to_csv(save_path, index=False)

    return games
