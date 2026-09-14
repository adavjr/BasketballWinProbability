"""
End to End run : Load data -> Build feature tables -> Train and evaluate models
"""

import gc
import pandas as pd
from DataLoad import load_nba_data, load_wbb_data, SEASONS
from FeatureBuilding import prepare_pbp, extract_game_results, build_features
from Elo import build_elo_ratings
from Model import train_and_evaluate, summarize
from Calibration_Plot import plot_reliability_diagrams

LOADERS = {
    "NBA": load_nba_data,
    "NCAA_WBB": load_wbb_data,
}

def run_league_across_seasons(league: str, seasons: list[int]) -> pd.DataFrame:
    """
    Processes one league across all seasons, season by season, 
    threading Elo rating state (and last-season-seen state, for the season-transition regression)
    forward from each season to the next.
    """
    loader = LOADERS[league]

    elo_ratings_state, elo_last_season_state = {}, {}
    season_feature_tables = []

    for season in seasons:
        print(f"  [{league}] loading season {season}...")
        raw = loader(seasons=[season])
        slim = prepare_pbp(raw)
        del raw
        gc.collect()

        game_results = extract_game_results(slim)
        elo_per_game, elo_ratings_state, elo_last_season_state = build_elo_ratings(
            game_results,
            initial_ratings=elo_ratings_state,
            initial_last_season_seen=elo_last_season_state,
        )

        feats = build_features(slim, league=league, elo_per_game=elo_per_game)
        del slim
        gc.collect()

        print(f"  [{league}] season {season}: {len(feats):,} feature rows, "
              f"{feats['game_id'].nunique():,} games")
        season_feature_tables.append(feats)
        
    return pd.concat(season_feature_tables, ignore_index=True)


def main():
    all_league_features = {}
    for league, seasons in [("NBA", SEASONS), ("NCAA_WBB", SEASONS)]:
        print(f"Processing {league} data from sportsdataverse...")
        all_league_features[league] = run_league_across_seasons(league, seasons)
    
    print("\nTraining models...")
    all_results = []
    for league, feats in all_league_features.items():
        all_results.extend(train_and_evaluate(feats, league=league))

    summary = summarize(all_results)
    print("\n=== Model performance (AUC + Brier score), pooled across seasons ===")
    print(summary.to_string(index=False))

    print("\n=== Feature importance comparison (XGBoost) ===")
    for r in all_results:
        if r.model_name == "XGBoost":
            print(f"\n{r.league}:")
            print(r.feature_importance.sort_values(ascending=False).to_string())

    summary.to_csv("outputs/model_summary_full.csv", index=False)
    print("\nPlotting reliability diagrams...")
    plot_reliability_diagrams(all_results, save_path="outputs/reliability_diagrams_full.png")
    print("\nSaved outputs/model_summary_full.csv and outputs/reliability_diagrams_full.png")


if __name__ == "__main__":
    main()