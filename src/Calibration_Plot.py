"""
Renders diagrams for model reliability (calibration) from the ModelResult objects in Model.train_and_evaluate().
A perfectly calibrated model's curve sits on the y = x linear. 
Points above it mean the model is underconfident in that probability range
Points below mean it's overconfident.
"""

import matplotlib.pyplot as plt
from Model import ModelResult

def plot_reliability_diagrams(results: list[ModelResult], save_path: str = "outputs/reliability_diagrams.png"):
    """
    One subplot per league, for both models (LogisticRegression, XGBoost)
    overlaid on same axes so they're directly comparable.
    """
    leagues = sorted(set(r.league for r in results))
    fig, axes = plt.subplots(1, len(leagues), figsize=(6 * len(leagues), 5.5), squeeze=False)
    axes = axes[0]

    for ax, league in zip(axes, leagues):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

        league_results = [r for r in results if r.league == league]
        for r in league_results:
            prob_true, prob_pred = r.calibration
            ax.plot(
                prob_pred, prob_true, marker="o", linewidth=2,
                label=f"{r.model_name} (Brier={r.brier:.3f}, AUC={r.auc:.3f})",
            )

        ax.set_title(f"{league} — Reliability Diagram")
        ax.set_xlabel("Predicted win probability")
        ax.set_ylabel("Observed win frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    return fig