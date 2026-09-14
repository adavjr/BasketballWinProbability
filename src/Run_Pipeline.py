"""
End to End run : Load data -> Build feature tables -> Train and evaluate models
"""

import pandas as pd
from DataLoad import load_both
from FeatureBuilding import build_features
from Model import train_and_evaluate, summarize
from Calibration_Plot import plot_reliability_diagrams

def main():
    print("Loading play-by-play data from sportsdataverse...")
    raw = load_both()

    print("Building features for model training...")
    nba_feats = build_features(raw["nba"], league="NBA")
    wbb_feats = build_features(raw["wbb"], league="NCAA_WBB")

    print(f"NBA: {len(nba_feats):,} feature rows across {nba_feats['game_id'].nunique():,} games")
    print(f"WBB: {len(wbb_feats):,} feature rows across {wbb_feats['game_id'].nunique():,} games")

    print("Training NBA models...")
    nba_results = train_and_evaluate(nba_feats, league="NBA")

    print("Training NCAA WBB models...")
    wbb_results = train_and_evaluate(wbb_feats, league="NCAA_WBB")

    all_results = nba_results + wbb_results
    summary = summarize(all_results)
    print("\n=== Model performance (AUC + Brier score) ===")
    print(summary.to_string(index=False))

    print("\n=== Feature importance comparison (XGBoost) ===")
    for r in all_results:
        if r.model_name == "XGBoost":
            print(f"\n{r.league}:")
            print(r.feature_importance.sort_values(ascending=False).to_string())

    summary.to_csv("outputs/model_summary.csv", index=False)
    print("\nSaved outputs/model_summary.csv")

    print("\nPlotting reliability diagrams...")
    plot_reliability_diagrams(all_results)

if __name__ == "__main__":
    main()