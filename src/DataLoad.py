"""
Pull play-by-play data for NBA and NCAA WBB via sportsdataverse 2023-2026 seasons
"""

import sportsdataverse as sdv
import pandas as pd
from pathlib import Path

SEASONS = [2024, 2025, 2026]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def load_nba_data(seasons: list[int] = SEASONS, force_refresh: bool = False) -> pd.DataFrame:
    cached_path = DATA_DIR / f"nba_pbp_{min(seasons)}_{max(seasons)}.parquet"

    if cached_path.exists() and not force_refresh:
        return pd.read_parquet(cached_path)

    df = sdv.load_nba_pbp(seasons=seasons, return_as_pandas=True)
    df["league"] = "NBA"
    df.to_parquet(cached_path, index=False)
    return df

def load_wbb_data(seasons: list[int] = SEASONS, force_refresh: bool = False) -> pd.DataFrame:
    cached_path = DATA_DIR / f"ncaa_wbb_pbp_{min(seasons)}_{max(seasons)}.parquet"

    if cached_path.exists() and not force_refresh:
        return pd.read_parquet(cached_path)

    df = sdv.load_wbb_pbp(seasons=seasons, return_as_pandas=True)
    df["league"] = "NCAA_WBB"
    df.to_parquet(cached_path, index=False)
    return df

def load_both(seasons: list[int] = SEASONS, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {
        "nba" : load_nba_data(seasons, force_refresh),
        "wbb" : load_wbb_data(seasons, force_refresh)
    }

if __name__ == "__main__":
    data = load_both()
    for league, df in data.items():
        print(f"{league}: {df.shape[0]:,} rows, {df['game_id'].nunique():,} games")