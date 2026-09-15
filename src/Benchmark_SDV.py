"""
Comparing Elo-based win probability model to SDV package win probability model.

One key detail is that the SDV package model requires a pregame home win probability in the input args,
as it only models how the probability changes play-by-play and does not derive team strength.
Given this, this is just comparing if the shape of the in-game probability curve is similar between the two models,
given the same starting point, not whether the SDV package model is better or worse than the Elo-based model.

The Elo-based pregame probability is fed into the SDV package model, so any difference in the resulting curves can be
attributed to model approaches of in-game probability evolution, not a difference in pregame metrics.

NOTE: Seeding Strategy
Two supporte ways to seed sportsdataverse's model (see seed_source
below): Elo-derived pregame probability ("elo", the default), or 
xgb model's play-0 prediction ("model"). The "elo" mode compares "xgb model
vs. sdv package model" including any disagreement about the pregame prior
itself. The "model" mode removes that confound by starting both curves
from the identical point, isolating disagreement to the in-game
EVOLUTION logic only. See Run_Benchmark.py's SEED_SOURCE setting and the
project write-up for why this distinction mattered in practice: an
initial "elo" comparison showed a persistent ~20-25 point gap between
the two curves for most of a game despite near-identical SHAPE, which
turned out to be explained by the XGBoost model's learned pregame
mapping disagreeing with the raw Elo formula — not by any disagreement
about how the game actually evolved.
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

def pregame_prob_from_model(model, feature_cols: list[str], game_features: pd.DataFrame) -> float:
    """
    Model's predicted probability at play index 0 (before anything
    has happened), rather than the raw Elo formula. Sorting by
    seconds_remaining descending guarantees the true first play is grabbed
    even if the input rows aren't already in chronological order.
 
    This is the seed to use when isolating the in-game EVOLUTION logic —
    see the SEEDING STRATEGY note at the top of this file for why the two
    seeding choices answer different questions.
    """
    first_play = game_features.sort_values("seconds_remaining", ascending=False).iloc[[0]]
    return model.predict_proba(first_play[feature_cols])[:, 1][0]


def _compare_single_game(game_id, season: int, league: str, elo_per_game: pd.DataFrame, model, 
                         feature_cols: list[str], game_features: pd.DataFrame,
                         seed_source: str = "elo") -> pd.DataFrame:
    """
    Shared comparison logic for NBA and NCAA WBB

    seed_source controls what pregame probability sportsdataverse's model
    is seeded with:
        "elo"   - the raw Elo logistic formula (pregame_prob_from_elo).
                  This is "xgb model vs. sdv package model" including any disagreement
                  about the pregame prior itself.
        "model" - xgb model's play-0 prediction (pregame_prob_from_model).
                  This isolates the in-game EVOLUTION logic only, since both
                  curves now start from the identical point.
    """
    if seed_source == "elo":
        pregame_prob = pregame_prob_from_elo(elo_per_game, game_id)
    elif seed_source == "model":
        pregame_prob = pregame_prob_from_model(model, feature_cols, game_features)
    else:
        raise ValueError(f"Unknown seed_source: {seed_source!r} (expected 'elo' or 'model')")

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
                         feature_cols: list[str], game_features: pd.DataFrame, seed_source: str = "elo") -> pd.DataFrame:
    """
    Args:
        game_id: game_id to benchmark
        season: the season game_id belongs to, identifies which raw file to pull.
        elo_per_game: this season's Elo pregame ratings (indexed by
            game_id), as returned by Elo.build_elo_ratings().
        model: a trained model from Model.train_and_evaluate() (or
            refit directly) with a .predict_proba() method.
        feature_cols: the feature column order model expects.
        game_featres: rows of feature table
            (FeatureBuilding.build_features output) for this game_id, in
            the same play order as the raw play-by-play.
        seed_source: "elo" (default) or "model" — see _compare_single_game's
            docstring for what each one isolates.
 
    Returns:
        A DataFrame with both models' per-play home win probability,
        aligned positionally
    """
    return _compare_single_game(game_id, season, "NBA", elo_per_game, model,
                                 feature_cols, game_features, seed_source=seed_source)

def _compare_single_wbb_game(game_id, season: int, elo_per_game: pd.DataFrame, model, 
                         feature_cols: list[str], game_features: pd.DataFrame, seed_source: str = "elo") -> pd.DataFrame:
    """
    Equivalent to _compare_single_nba_game but for NCAA WBB - reference that docstring
    """
    return _compare_single_game(game_id, season, "NCAA_WBB", elo_per_game, model,
                                 feature_cols, game_features, seed_source=seed_source)

def plot_comparison(comparison: pd.DataFrame, game_id: int, save_path: str = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(comparison["play_index"], comparison["sportsdataverse_home_wp"], 
            label="sportsdataverse package model", linewidth=2)
    ax.plot(comparison["play_index"], comparison["model_home_wp"], 
                label="Elo + XGBoost model", linewidth=2, alpha=0.8)
    ax.set_title(f"Home win probability over the game - game_id {game_id}")
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
        "Refer to README for an example, instead of running this file on its own or run 'Run_Benchmark.py'"
    )