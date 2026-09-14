"""
Train and evaluate win-probability models for given league feature table (output from features.build_features).
Two models are trained per league:
1. Logistic regression : interpretable coefficients
2. XGBoost : handles nonlinearity natively

Model evaluation includes discrimination (AUC) and calibration (Brier score + reliability curve),
since a 70% win probability is only useful if it actually wins ~70% of the time.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb

FEATURE_COLS = [
    "score_diff", "frac_game_remaining", "diff_per_time_pressure",
    "home_timeouts_used", "away_timeouts_used", "strength_diff",
]

@dataclass
class ModelResult:
    league: str
    model_name: str
    auc: float
    brier: float
    calibration : tuple #from sklearn.calibration_curve
    feature_importance: pd.Series
    fitted_model: object #actual fitted sklearn/xgboost estimator

def _train_test_split_by_game(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Split by game_id, to avoid plays leaking across train/test. 
    A random row-level split would inflate AUC and calibration, 
    because rows from the same game are highly correlated.
    """

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["game_id"]))
    return df.iloc[train_idx], df.iloc[test_idx]

def train_and_evaluate(df: pd.DataFrame, league: str) -> list[ModelResult]:
    train_df, test_df = _train_test_split_by_game(df)
    X_train, y_train = train_df[FEATURE_COLS], train_df["home_win"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["home_win"]

    results = []

    # -- Logistic Regression --
    logit = LogisticRegression(max_iter=1000)
    logit.fit(X_train, y_train)
    probs_logit = logit.predict_proba(X_test)[:, 1]
    results.append(ModelResult(
        league=league, model_name="LogisticRegression",
        auc=roc_auc_score(y_test, probs_logit),
        brier=brier_score_loss(y_test, probs_logit),
        calibration=calibration_curve(y_test, probs_logit, n_bins=10),
        feature_importance=pd.Series(logit.coef_[0], index=FEATURE_COLS).sort_values(),
        fitted_model=logit,
    ))

    # -- XGBoost --
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    probs_xgb = xgb_model.predict_proba(X_test)[:, 1]
    results.append(ModelResult(
        league=league, model_name="XGBoost",
        auc=roc_auc_score(y_test, probs_xgb),
        brier=brier_score_loss(y_test, probs_xgb),
        calibration=calibration_curve(y_test, probs_xgb, n_bins=10),
        feature_importance=pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS).sort_values(),
        fitted_model=xgb_model,
    ))

    return results

def summarize(results: list[ModelResult]) -> pd.DataFrame:
    """
    Summarize model results in a DataFrame for easy comparison.
    """
    return pd.DataFrame([
        {"league": r.league, "model": r.model_name, "auc": round(r.auc, 4), "brier": round(r.brier, 4)}
        for r in results
    ])