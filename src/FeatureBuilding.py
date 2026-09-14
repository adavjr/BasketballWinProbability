"""
Build win-probability feature set from raw play-by-play data. For now, NBA and WBB share
key column names, so this can be done with one function, and separation will take
place when the models are trained
"""

import numpy as np
import pandas as pd

TOTAL_GAME_SECONDS = {
    "NBA" : 48 * 60,     #12-min quarters
    "NCAA_WBB" : 40 * 60      #10-min quarters
}

EXHIBITIONS = {"CHK", "SHQ", "WLD", "USA", "WORLD"} #All-Star / exhibition rosters, no real season record

def prepare_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Slim raw play-by-play down to only the columns needed downstream, and
    drop exhibition/all-star games. Keeping this separate means both
    extract_game_results() and build_features() share one cleaning step
    instead of duplicating the logic.
    """
    needed_cols = [
        "game_id", "season", "game_date",
        "home_team_abbrev", "away_team_abbrev",
        "home_score", "away_score",
        "start_game_seconds_remaining",
    ]
    optional_cols = ["home_timeout_called", "away_timeout_called"]
    present_optional = [c for c in optional_cols if c in pbp.columns]
    df = pbp[needed_cols + present_optional].copy()

    df["home_team_abbrev"] = df["home_team_abbrev"].astype("category")
    df["away_team_abbrev"] = df["away_team_abbrev"].astype("category")

    df = df[
        ~df["home_team_abbrev"].isin(EXHIBITIONS)
        & ~df["away_team_abbrev"].isin(EXHIBITIONS)
    ]

    #Per-game final score, via groupby().transform() rather than a
    #sort + merge of the full frame 
    df["final_home_score"] = df.groupby("game_id")["home_score"].transform("last")
    df["final_away_score"] = df.groupby("game_id")["away_score"].transform("last")
    df["home_win"] = (df["final_home_score"] > df["final_away_score"]).astype(int)

    return df

def extract_game_results(df_slim: pd.DataFrame) -> pd.DataFrame:
    """
    One row per game_id — the minimal table build_elo_ratings() needs.
    Orders of magnitude smaller than df_slim, so this is cheap even
    though df_slim itself may be a full season of play-by-play.
    """
    return (
        df_slim[["game_id", "season", "game_date", "home_team_abbrev", "away_team_abbrev",
                 "final_home_score", "final_away_score"]]
        .drop_duplicates("game_id")
        .rename(columns={"final_home_score": "home_score", "final_away_score": "away_score"})
    )
def build_features(df_slim: pd.DataFrame, league: str, elo_per_game: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for win-probability model training. For now, NBA and WBB share
    key column names, so this can be done with one function, and separation will take
    place when the models are trained
    """

    total_secs = TOTAL_GAME_SECONDS[league]
    df = df_slim.copy()

    df["score_diff"] = df["home_score"] - df["away_score"]
    df["seconds_remaining"] = df["start_game_seconds_remaining"].clip(lower=0)
    df["frac_game_remaining"] = (df["seconds_remaining"] / total_secs).clip(0, 1)
    df["diff_per_time_pressure"] = df["score_diff"] / np.sqrt(df["seconds_remaining"] + 1)

    if "home_timeout_called" in df.columns:
        df["home_timeouts_used"] = df["home_timeout_called"].astype(int)
        df["away_timeouts_used"] = df["away_timeout_called"].astype(int)
    else:
        df["home_timeouts_used"] = 0
        df["away_timeouts_used"] = 0

    df = df.merge(elo_per_game, left_on="game_id", right_index=True, how="left")
    df["strength_diff"] = df["home_pregame_elo"] - df["away_pregame_elo"]

    feature_cols = [
        "score_diff", "frac_game_remaining", "diff_per_time_pressure",
        "home_timeouts_used", "away_timeouts_used", "strength_diff",
    ]
    # Context columns aren't used for training (model.py selects FEATURE_COLS
    # explicitly) but travel along so downstream consumers — like the
    # Streamlit app's export step — never need to re-derive or re-merge
    # anything to reconstruct a human-readable game replay. Keeping this in
    # the SAME dataframe that gets predict_proba'd avoids a row-alignment
    # bug where a separately-built "display" frame and the "model input"
    # frame could silently drift apart after dropna().
    context_cols = [
        "home_team_abbrev", "away_team_abbrev", "home_score", "away_score",
        "seconds_remaining",
    ]
    keep_cols = feature_cols + context_cols + ["home_win", "game_id", "season"]
    result = df[keep_cols].dropna(subset=feature_cols).reset_index(drop=True)
    result["league"] = league
    return result

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