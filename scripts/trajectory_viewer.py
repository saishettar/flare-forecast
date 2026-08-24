"""Trajectory viewer: predicted vs. actual activity score over real timelines.

Uses score_regression, the model that actually won the comparison in
train_forecast.py, not a microbiome model. Plotting the microbiome
models here would dress up a negative result as something more
interesting than it is.

For each diagnosis, picks three subjects out of everyone with at least
4 leave-one-subject-out forecast predictions: the one the model did
best on, the one it did worst on, and the one closest to the median.
Showing only the best case would be a nicer-looking plot and a
misleading one, so all three go in the same figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flare_forecast.data import DIAGNOSIS_SCORE_COL, _load_scored_mgx, build_forecast_dataset  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "trajectories"
MIN_PAIRS_PER_SUBJECT = 4


def loso_predictions_with_metadata(diagnosis: str) -> pd.DataFrame:
    """Same model as train_forecast.py's score_regression, but keeping
    Participant ID and the predicted timepoint's real week_num, which the
    R^2-only version in train_forecast.py throws away."""
    X_t, score_t, y, groups, gap, week_t = build_forecast_dataset(diagnosis)
    score_t, y = score_t.to_numpy(), y.to_numpy()
    groups, gap, week_t = groups.to_numpy(), gap.to_numpy(), week_t.to_numpy()

    logo = LeaveOneGroupOut()
    rows = []
    for train_idx, test_idx in logo.split(score_t.reshape(-1, 1), y, groups):
        model = LinearRegression()
        model.fit(score_t[train_idx].reshape(-1, 1), y[train_idx])
        preds = model.predict(score_t[test_idx].reshape(-1, 1))
        for idx, pred in zip(test_idx, preds):
            rows.append({
                "Participant ID": groups[idx],
                "target_week": week_t[idx] + gap[idx],
                "y_true": y[idx],
                "y_pred": pred,
            })
    return pd.DataFrame(rows)


def pick_representative_subjects(preds: pd.DataFrame) -> dict[str, str]:
    """Best/median/worst subject by mean absolute error, among subjects
    with enough predictions for the plot to actually show a trajectory."""
    preds = preds.assign(abs_err=(preds["y_true"] - preds["y_pred"]).abs())
    per_subject = preds.groupby("Participant ID").agg(n=("abs_err", "size"), mae=("abs_err", "mean"))
    eligible = per_subject[per_subject["n"] >= MIN_PAIRS_PER_SUBJECT].sort_values("mae")
    if len(eligible) < 3:
        eligible = per_subject.sort_values("mae")  # fall back if too few qualify
    best = eligible.index[0]
    worst = eligible.index[-1]
    median = eligible.index[len(eligible) // 2]
    return {"best": best, "median": median, "worst": worst}


def plot_diagnosis(diagnosis: str) -> Path:
    score_col = DIAGNOSIS_SCORE_COL[diagnosis]
    scored = _load_scored_mgx(diagnosis)  # every real visit, not just forecast pairs
    preds = loso_predictions_with_metadata(diagnosis)
    subjects = pick_representative_subjects(preds)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)
    for ax, (label, pid) in zip(axes, subjects.items()):
        actual = scored[scored["Participant ID"] == pid].sort_values("week_num")
        pred = preds[preds["Participant ID"] == pid].sort_values("target_week")
        pred_agg = pred.groupby("target_week")["y_pred"].mean()  # average multi-source hits on one target week

        ax.plot(actual["week_num"], actual["score"], "o-", color="#333333",
                 label=f"actual {score_col}", linewidth=1.5, markersize=4)
        ax.plot(pred_agg.index, pred_agg.values, "x--", color="#c0392b",
                 label="predicted (score_regression, LOSO)", linewidth=1.5, markersize=7)
        mae_val = (pred["y_true"] - pred["y_pred"]).abs().mean()
        ax.set_title(f"{label} case: subject {pid} (n={len(pred)} predictions, MAE={mae_val:.2f})")
        ax.set_xlabel("week")
        ax.set_ylabel(score_col.upper())
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"{diagnosis}: predicted vs. actual {score_col.upper()}, held-out subjects")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{diagnosis.lower()}_trajectories.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    for diagnosis in ("CD", "UC"):
        path = plot_diagnosis(diagnosis)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
