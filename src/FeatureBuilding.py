"""
Build win-probability feature set from raw play-by-play data. For now, NBA and WBB share
key column names, so this can be done with one function, and separation will take
place when the models are trained
"""

import numpy as np
import pandas as pd

TOTAL_GAME_SECONDS = {
    "NBA" : 48 * 60,     #12-min quarters
    "WBB" : 40 * 60      #10-min quarters
}

def build_team_strength(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Simple point-differential rating per team (season-to-date), to serve as pre-game strength
    signal. Will be updated to a more complex ELO rating
    """

    game_final = (
        pbp.sort_values("start_game_seconds_remaining", ascending=False)
        .groupby("game_id")
        .first()[["season", "home_team_abbrev", "away_team_abbrev", "home_score", "away_score", "game_date"]]
        .reset_index()
    )

    records = []
    for _, row in game_final.iterrows():
        margin = row["home_score"] - row["away_score"]
        records.append({"season" : row["season"], "team" : row["home_team_abbrev"],
                         "game_date" : row["game_date"], "margin": margin})
        records.append({"season" : row["season"], "team" : row["away_team_abbrev"],
                         "game_date" : row["game_date"], "margin": -margin})

    long_df = pd.DataFrame(records).sort_values(["team", "season", "game_date"])
    #Expanding mean margin, per team per season, computed before game
    #Utilizes shift(1) to avoid leakage of current game into feature

    long_df["pregame_avg_margin"] = (
        long_df.groupby(["team", "season"])["margin"]
        .transform(lambda s : s.shift(1).expanding().mean())
        .fillna(0.0)
    )

    return long_df[["team", "season", "game_date", "pregame_avg_margin"]]