"""
Interactive dashboard to view win-probability. Only reads statistical
artifacts produced by src/export_app_artifacts.py. This file will never
call the model or the sportsdataverse api directly, keeping it efficient.

Run : streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

st.set_page_config(page_title="Basketball Win Probability", layout="wide")

@st.cache_data
def load_artifacts():
    games = pd.read_parquet(ARTIFACT_DIR / "games_index.parquet")
    curves = pd.read_parquet(ARTIFACT_DIR / "wp_curves.parquet")
    calibration = pd.read_parquet(ARTIFACT_DIR / "calibration_data.parquet")

    with open(ARTIFACT_DIR / "model_metrics.json") as f:
        metrics = pd.DataFrame(json.load(f))

    return games, curves, calibration, metrics

def format_clock(seconds_remaining: float, league: str) -> str:
    """ Turn seconds-remaining-in-game value back into readable game clock"""
    period_length = 12 * 60 if league == "NBA" else 10 * 60
    # Seconds_remaining counts down for whole game, so have to separate by 
    # quarter and time remaining in quarter
    total_periods = 4
    total_game_seconds = period_length * total_periods
    elapsed = total_game_seconds - seconds_remaining
    period = min(int(elapsed // period_length) + 1, total_periods)
    secs_into_period = elapsed - (period - 1) * period_length
    secs_left_in_period = max(period_length - secs_into_period, 0)
    mm, ss = divmod(int(secs_left_in_period), 60)

    return f"Q{period} {mm:02d}:{ss:02d}"

games, curves, calibration, metrics = load_artifacts()

st.title("NBA & NCAA Women's Basketball - Win Probability")

tab_replay, tab_diagnostics = st.tabs(["Game Replay", "Model Diagnostics"])

with tab_replay:
    col_league, col_season, col_game = st.columns([1, 1, 2])

    with col_league:
        league = st.selectbox("League", sorted(games["league"].unique()))

    league_games = games[games["league"] == league]

    with col_season:
        season = st.selectbox("Season", sorted(league_games["season"].unique(), reverse=True))

    season_games = league_games[league_games["season"] == season].sort_values("game_date", ascending=False)
    season_games = season_games.assign(
        label = lambda d: d["game_date"].astype(str) + "  -  "
        + d["home_team_abbrev"] + " vs " + d["away_team_abbrev"]
        + " (" + d["final_home_score"].astype(str) + "-" + d["final_away_score"].astype(str) + ")"
    )

    with col_game:
        game_label = st.selectbox("Game", season_games["label"])

    selected_game_id = season_games.loc[season_games["label"] == game_label, "game_id"].iloc[0]
    game_curve = curves[curves["game_id"] == selected_game_id].reset_index(drop=True)
    game_curve = game_curve.sort_values("seconds_remaining", ascending=False).reset_index(drop=True)

    game_meta = season_games[season_games["game_id"] == selected_game_id].iloc[0]

    st.subheader(
        f"{game_meta['home_team_abbrev']} {game_meta['final_home_score']}"
        f"- {game_meta['final_away_score']} {game_meta['away_team_abbrev']}"
    )

    #Scrubber feature: index into play sequence, not seconds, since plays aren't evenly time spaced
    play_idx = st.slider("Scrub through the game", 0, len(game_curve) - 1, len(game_curve) - 1)
    current = game_curve.iloc[play_idx]

    m1, m2, m3 = st.columns(3)
    m1.metric("Score", f"{int(current['home_score'])} - {int(current['away_score'])}")
    m2.metric("Clock", format_clock(current["seconds_remaining"], league))
    m3.metric(f"{game_meta['home_team_abbrev']} win probability", f"{current['home_win_prob']:.1%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(game_curve))), y=game_curve["home_win_prob"], mode="lines",
        name=f"{game_meta['home_team_abbrev']} win probability", line=dict(width=2)
    ))
    fig.add_vline(x=play_idx, line_dash="dash", line_color="gray")
    fig.add_hline(y=0.5, line_dash="dot", line_color="lightgray")
    fig.update_layout(
        xaxis_title="Play (chronological)", yaxis_title="Home win probability",
        yaxis_range=[0,1], height=420, margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Win probability is determined by Elo-based XGBoost model's estimate at each play,"
        "not sportsdataverse's package model - see benchmark files for comparison"
    )
    