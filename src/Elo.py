"""
Computes Elo ratings as a pre-game strength metric per team, per game; replaces earlier expanding-mean-margin approach. See features.py for how these ratings are merged back onto the full play-by-play frame.

This loops over each game instead of the individual pbp rows, and the resulting per-game pregame ratings
are meged back onto the full pbp frame in FeatureBuilding.py. 
This is because Elo is inherently sequential (each game's rating update depends on the state left by the previous game),
so it is more efficient to loop over the thousands of games rather than the millions of pbp rows.

Adapted from FiveThirtyEight's published NBA Elo approach:
https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/
  - Home-court advantage is added to the home team's rating before computing expected outcome
  - A margin-of-victory multiplier means blowouts move ratings more than narrow wins.
  - Ratings partially regress toward the mean at the start of each new season, since rosters turn over. 
        NCAA WBB in particular sees far more year-to-year roster turnover than the NBA, 
        so this matters more there than in the pros — worth explicitly noting instead of assuming one constant fits both leagues equally well.
"""

import pandas as pd

BASE_RATING = 1500.0
HOME_ADVANTAGE = 100.0   # Elo points added to home team's rating pre-game
K_FACTOR = 20.0
SEASON_REGRESSION = 1 / 3  # fraction of each team's rating pulled back towards BASE_RATING at season start

def _expected_home_win_prob(home_rating: float, away_rating: float) -> float:
    elo_diff = (home_rating + HOME_ADVANTAGE) - away_rating
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400))

def _margin_of_victory_multiplier(point_margin: float, elo_diff: float) -> float:
    """
    538's NBA Elo margin-of-victory multiplier: blowouts move ratings more than close games,
    but with diminishing returns, and it's dampened when the favorite (by elo_diff) wins by a lot (since that's less surprising).
    """
    return ((abs(point_margin) + 3) ** 0.8) / (7.5 + 0.006 * abs(elo_diff))

def build_elo_ratings(games: pd.DataFrame, initial_ratings: dict = None,
                       initial_last_season_seen: dict = None):
    """
    Args:
        games: one row per game, columns: game_id, season, game_date,
               home_team_abbrev, away_team_abbrev, home_score, away_score.
               Must already have exhibition games excluded.
        initial_ratings: team -> rating dict to start from (e.g. the final
               state returned after processing a prior season). None starts
               every team at BASE_RATING.
        initial_last_season_seen: team -> season dict, paired with
               initial_ratings, so this call knows whether a team's carried
               rating is from last season or earlier (and thus needs to be
               regressed toward the mean).
    Returns:
        elo_per_game: DataFrame indexed by game_id with columns:
            home_pregame_elo, away_pregame_elo, home_postgame_elo,
            away_postgame_elo.
        final_ratings: team -> rating dict after processing all games.
        final_last_season_seen: team -> season dict after processing all games.
    """
    games = games.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)

    ratings = dict(initial_ratings) if initial_ratings else {}
    last_season_seen = dict(initial_last_season_seen) if initial_last_season_seen else {}
    records = []

    for row in games.itertuples(index=False):
        home, away, season = row.home_team_abbrev, row.away_team_abbrev, row.season

        # Regress each team's carried-over rating toward the mean
        # if this is their first game of the season,
        # ratings.setdefault handles team's first ever appearance
        for team in (home, away):
            if team in ratings and last_season_seen.get(team) != season:
                ratings[team] = (
                    (1-SEASON_REGRESSION) * ratings[team] + SEASON_REGRESSION * BASE_RATING
                )
            last_season_seen[team] = season

        home_rating = ratings.setdefault(home, BASE_RATING)
        away_rating = ratings.setdefault(away, BASE_RATING)

        records.append({
            "game_id": row.game_id,
            "home_pregame_elo": home_rating,
            "away_pregame_elo": away_rating,
        })

        #Update ratings using game's actual result
        margin = row.home_score - row.away_score
        elo_diff = (home_rating + HOME_ADVANTAGE) - away_rating
        expected_home_win_prob = _expected_home_win_prob(home_rating, away_rating)
        actual_home_win = 1.0 if margin > 0 else 0.0
        mov_multiplier = _margin_of_victory_multiplier(margin, elo_diff)

        # Update ratings based on game outcome
        actual_home_win = 1 if margin > 0 else 0
        delta = K_FACTOR * mov_multiplier * (actual_home_win - expected_home_win_prob)

        ratings[home] = home_rating +delta
        ratings[away] = away_rating - delta

        per_game_df = pd.DataFrame(records).set_index("game_id")
        return per_game_df, ratings, last_season_seen
