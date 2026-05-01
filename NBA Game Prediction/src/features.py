import pandas as pd
import os

def create_features(df, window=5, save_path="data/processed/games.csv"):
    """
    Create leakage-free rolling features using ONLY past games.
    """
    df = df.copy()
    df = df.sort_values("GAME_DATE")

    stats = ['PTS', 'REB', 'AST', 'TOV', 'FG_PCT']

    for stat in stats:
        df[f'{stat}_rolling'] = (
            df.groupby('TEAM_ID')[stat]
              .transform(lambda x: x.shift(1).rolling(window, min_periods=window).mean())
        )

    # Drop rows with insufficient history
    df = df.dropna()

    # Save processed data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    feature_cols = [f'{stat}_rolling' for stat in stats]

    return df, feature_cols
