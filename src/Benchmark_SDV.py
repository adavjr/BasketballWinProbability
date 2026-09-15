"""
Comparing Elo-based win probability model to SDV package win probability model.

One key detail is that the SDV package model requires a pregame home win probability in the input args,
as it only models how the probability changes play-by-play and does not derive team strength.
Given this, this is just comparing if the shape of the in-game probability curve is similar between the two models,
given the same starting point, not whether the SDV package model is better or worse than the Elo-based model.

The Elo-based pregame probability is fed into the SDV package model, so any difference in the resulting curves can be
attributed to model approaches of in-game probability evolution, not a difference in pregame metrics. 
"""

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

from sportsdataverse.nba.nba_game_predict import nba_in_game_win_prob
from sportsdataverse.nba.nba_loaders import load_nba_pbp
from sportsdataverse.wbb.wbb_game_predict import wbb_in_game_win_prob
from sportsdataverse.wbb.wbb_loaders import load_wbb_pbp

from Elo import _expected_home_win_prob

def pregame_prob_from_elo(elo_per_game: pd.DataFrame, game_id: int) -> float:
    """Convert stored pregame Elo ratings for a game into a win probability"""
    row = elo_per_game.loc[game_id]
    return _expected_home_win_prob(row["home_pregame_elo"], row["away_pregame_elo"])

def _compare_single_game(game_id, season: int, league: str, elo_per_game: pd.DataFrame, model, 
                         feature_cols: list[str], game_features: pd.DataFrame) -> pd.DataFrame:
    """
    Shared comparison logic for NBA and NCAA WBB
    """
    pregame_prob = pregame_prob_from_elo(elo_per_game, game_id)

    if league == "NBA":
        raw_pbp = load_nba_pbp([season]).filter(pl.col("game_id") == game_id)
        sdv_wp = nba_in_game_win_prob(raw_pbp, pregame_prob, return_as_pandas=True)
    elif league == "NCAA_WBB":
        raw_pbp = load_wbb_pbp([season]).filter(pl.col("game_id") == game_id)
        sdv_wp = wbb_in_game_win_prob(raw_pbp, pregame_prob, return_as_pandas=True)
    else:
        raise ValueError(f"Unknown league: {league}")

    model_prob = model.predict_proba(game_features[feature_cols])[:, 1]

    n = min(len(sdv_wp), len(model_prob))
    comparison = pd.DataFrame({
        "play_index" : range(n),
        "sportsdataverse_home_wp" : sdv_wp["home_win_prob"].values[:n],
        "model_home_wp" : model_prob[:n]
    })

    comparison["abs_diff"] = (comparison["sportsdataverse_home_wp"] - comparison["model_home_wp"]).abs()
    return comparison


def _compare_single_nba_game(game_id, season: int, elo_per_game: pd.DataFrame, model, 
                         feature_cols: list[str], game_features: pd.DataFrame) -> pd.DataFrame:
    return _compare_single_game(game_id, season, "NBA", elo_per_game, model, feature_cols, game_features)

def _compare_single_wbb_game(game_id, season: int, elo_per_game: pd.DataFrame, model, 
                         feature_cols: list[str], game_features: pd.DataFrame) -> pd.DataFrame:
    return _compare_single_game(game_id, season, "NCAA_WBB", elo_per_game, model, feature_cols, game_features)

def plot_comparison(comparison: pd.DataFrame, game_id: int, save_path: str = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(comparison["play_index"], comparison["sportsdataverse_home_wp"], 
            label="sportsdataverse package model", linewidth=2)
    ax.plot(comparison["play_index"], comparison["model_home_wp"], 
                label="Elo + XGBoost model", linewidth=2, alpha=0.8)
    ax.set_title(f"Home win probability over te game - game_id {game_id}")
    ax.set_xlabel("Play index (chronological)")
    ax.set_ylabel("Home win probability")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")

    return fig

if __name__ == "__main__":
    print(
        "This file has been designed for an already existing trained model, with the season's"
        "elo_per_game table stored in memory, produced by the Run_Pipeline.py file in this directory."
        "Refer to README for an example, instead of running this file on its own"
    )