"""
Standalone benchmark script: trains model on a single season, then benchmark's the play-by-play
win probability against sportsdataverse's packaged in-game model, across N randomly sampled games,
providing an aggregate distribution result.

"""

import random
import pandas as pd

from DataLoad import load_nba_data, load_wbb_data
from FeatureBuilding import prepare_pbp, extract_game_results, build_features
from Elo import build_elo_ratings
from Model import train_and_evaluate, FEATURE_COLS
from Benchmark_SDV import _compare_single_nba_game, _compare_single_wbb_game, plot_comparison

LEAUGE = "NBA"          #"NBA" or "NCAA_WBB"
SEASON = 2025           #Choose already cached season
N_GAMES = 25            #Number of random games to use in benchmark
RANDOM_SEED = None      #Set to any int to get same sample each run
SEED_SOURCE = "model"     #"elo" (raw Elo formula) or "model" (xgb mode's first play prediction)
                        # - see Benchmark_SDV.py docstring for what each source isolates


LOADERS = {"NBA" : load_nba_data, "NCAA_WBB" : load_wbb_data}
COMPARE_FUNCS = {"NBA" : _compare_single_nba_game, "NCAA_WBB" : _compare_single_wbb_game}


def main():
    loader = LOADERS[LEAUGE]
    compare_func = COMPARE_FUNCS[LEAUGE]

    raw = loader(seasons=[SEASON])
    slim = prepare_pbp(raw)
    game_results = extract_game_results(slim)
    elo_per_game, _, _ = build_elo_ratings(game_results)
    feats = build_features(slim, league=LEAUGE, elo_per_game=elo_per_game)

    results = train_and_evaluate(feats, league=LEAUGE)
    xgb_result = [r for r in results if r.model_name == "XGBoost"][0]

    #Sample without replacement, from unique game_ids (avoids biasing to games with more play-by-play rows)
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    all_game_ids = feats["game_id"].unique().tolist()
    sample_game_ids = random.sample(all_game_ids, k=min(N_GAMES, len(all_game_ids)))

    per_game_summaries = []
    all_comparisons = []

    for i, game_id in enumerate(sample_game_ids, start=1):
        game_feats = feats[feats["game_id"] == game_id]
        try:
            comparison = compare_func(
                game_id=game_id,
                season=SEASON,
                elo_per_game=elo_per_game,
                model=xgb_result.fitted_model,
                feature_cols=FEATURE_COLS,
                game_features=game_feats,
                seed_source=SEED_SOURCE
            )
        except Exception as e:
            #If any single game's data causes an error, skip and note in summary
            print(f" [{i}/{len(sample_game_ids)}] game_id {game_id}: SKIPPED {e}")
            continue

        comparison["game_id"] = game_id
        all_comparisons.append(comparison)
        per_game_summaries.append({
            "game_id" : game_id,
            "n_plays" : len(comparison),
            "mean_abs_diff" : comparison["abs_diff"].mean(),
            "max_abs_diff" : comparison["abs_diff"].max()
        })
        print(f" [{i}/{len(sample_game_ids)}] game_id {game_id}: "
              f"{len(comparison)} plays, mean abs diff = {comparison['abs_diff'].mean():.4f}")


    #Error for if every game failed to load
    if not per_game_summaries:
        raise RuntimeError(
            f"All {len(sample_game_ids)} sampled games failed to benchmark — see the "
            f"SKIPPED messages above for the actual error on each one. Nothing to aggregate."
        )


    per_game_df = pd.DataFrame(per_game_summaries)
    pooled_df = pd.concat(all_comparisons, ignore_index=True)

    print(f"\n== Aggregate results across {len(per_game_df)} games ==")
    print(f"Mean of per-game mean abs diff  : {per_game_df['mean_abs_diff'].mean():.4f}")
    print(f"Std of per-game mean abs diff   : {per_game_df['mean_abs_diff'].std():.4f}")
    print(f"Worst single-game mean abs diff : {per_game_df['mean_abs_diff'].max():.4f} "
            f"(game_id {per_game_df.loc[per_game_df['mean_abs_diff'].idxmax(), 'game_id']})")
    print(f"Pooled mean abs diff (every play, every game): {pooled_df['abs_diff'].mean():.4f}")

    per_game_df.to_csv(f"outputs/{SEED_SOURCE}_{LEAUGE}_benchmark_per_game_summary.csv", index=False)
    print(f"\nSaved outputs/{SEED_SOURCE}_{LEAUGE}_benchmark_per_game_summary.csv")

    #Plot single game with largest disagreement between models, most informative graphic
    worst_game_id = per_game_df.loc[per_game_df["mean_abs_diff"].idxmax(), "game_id"]
    worst_comparison = pooled_df[pooled_df["game_id"] == worst_game_id].reset_index(drop=True)
    plot_comparison(worst_comparison, worst_game_id, save_path=f"outputs/{SEED_SOURCE}_{LEAUGE}_benchmark_worst_game.png")
    print(f"\nSaved outputs/{SEED_SOURCE}_{LEAUGE}_benchmark_worst_game.png")


if __name__ == "__main__":
    main()

        