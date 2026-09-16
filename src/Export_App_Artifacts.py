"""
Runs the win-probability pipeline and writes small, static artifact files that the
Streamlit app reads at request time. The app never calls sportsdataverse
or refits a model directly — it only loads these files.

Run this once after any pipeline change:
    python src/export_app_artifacts.py

Artifacts written to app/artifacts/:
    games_index.parquet      - one row per game, for the game picker
    wp_curves.parquet        - one row per play, per game: our model's
                                predicted win probability over time
    model_metrics.json       - AUC/Brier per league/model
    calibration_data.parquet - reliability-diagram points per league/model
"""

import json
from pathlib import Path

import pandas as pd
from DataLoad import load_nba_data, load_wbb_data, SEASONS
from FeatureBuilding import prepare_pbp, extract_game_results, build_features
from Elo import build_elo_ratings
from Model import train_and_evaluate, FEATURE_COLS

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "app" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

LOADERS = {"NBA" : load_nba_data, "NCAA_WBB" : load_wbb_data}

def process_league(league: str, seasons: list[int]):
    loader = LOADERS[league]
    elo_state, elo_last_season = {}, {}
    season_feature_tables, season_game_rows = [], []

    for season in seasons:
        print(f" [{league}] season {season}...")
        raw = loader(seasons=[season])
        slim = prepare_pbp(raw)
        del raw

        game_results = extract_game_results(slim)
        elo_per_game, elo_state, elo_last_season = build_elo_ratings(
            game_results, initial_ratings=elo_state, initial_last_season_seen=elo_last_season
        )

        feats = build_features(slim, league=league, elo_per_game=elo_per_game)
        season_feature_tables.append(feats)
        season_game_rows.append(game_results.assign(league=league))

        del slim
        print(f"  {len(feats):,} rows, {feats['game_id'].nunique():,} games")

    all_feats = pd.concat(season_feature_tables, ignore_index=True)
    all_games = pd.concat(season_game_rows, ignore_index=True)
    return all_feats, all_games

def main():
    games_index_parts, curve_parts, metrics_rows, calibration_rows = [], [], [], []

    for league in ["NBA", "NCAA_WBB"]:
        print(f"Processing {league}...")
        feats, games = process_league(league, SEASONS)

        results = train_and_evaluate(feats, league=league)
        best = [r for r in results if r.model_name == "XGBoost"][0]

        feats = feats.reset_index(drop=True)
        feats["home_win_prob"] = best.fitted_model.predict_proba(feats[FEATURE_COLS])[:, 1]
        curve_parts.append(feats)

        games_index_parts.append(
            games.rename(columns={"home_score": "final_home_score", "away_score": "final_away_score"})
        )

        for r in results:
            metrics_rows.append({"league": r.league, "model": r.model_name, "auc": r.auc, "brier": r.brier})
            prob_true, prob_pred = r.calibration
            for pt, pp in zip(prob_true, prob_pred):
                calibration_rows.append({"league": r.league, "model": r.model_name, "prob_true": pt, "prob_pred": pp})

    pd.concat(games_index_parts, ignore_index=True).to_parquet(ARTIFACT_DIR / "games_index.parquet", index=False)
    pd.concat(curve_parts, ignore_index=True).to_parquet(ARTIFACT_DIR / "wp_curves.parquet", index=False)
    pd.DataFrame(calibration_rows).to_parquet(ARTIFACT_DIR / "calibration_data.parquet", index=False)
    with open(ARTIFACT_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics_rows, f, indent=2)

    print(f"\nArtifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()