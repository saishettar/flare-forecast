"""Phase 2 baseline: cross-sectional activity-score regression.

Predicts same-timepoint HBI (Crohn's) or SCCAI (UC) from species-level
gut microbiome composition, ElasticNet with a subject-grouped nested CV:

- Outer GroupKFold(5) by Participant ID gives the reported R² — this is
  the number that matters, and it must be subject-grouped because HMP2
  has up to 24 repeated-measures timepoints per subject; an ungrouped
  split would let the same patient's near-identical microbiome appear
  in both train and test.
- Inner GroupKFold(4), also by subject, does the alpha/l1_ratio search
  (via GridSearchCV) within each outer training fold, so hyperparameter
  selection doesn't leak test-fold subjects either.

This is the de-risking step before Phase 3's real target (forecasting
activity 2-4 weeks ahead) — same validation shape, no temporal element
yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flare_forecast.data import build_baseline_dataset  # noqa: E402
from flare_forecast.features import ArcsinSqrtTransform, PrevalenceFilter  # noqa: E402

PARAM_GRID = {
    "model__alpha": np.logspace(-3, 1, 9),
    "model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
}


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("prevalence", PrevalenceFilter(min_prevalence=0.1)),
        ("arcsin_sqrt", ArcsinSqrtTransform()),
        ("scale", StandardScaler()),
        ("model", ElasticNet(max_iter=10_000)),
    ])


def run(diagnosis: str, n_outer_splits: int = 5, n_inner_splits: int = 4) -> None:
    X, y, groups = build_baseline_dataset(diagnosis)
    X, y, groups = X.to_numpy(), y.to_numpy(), groups.to_numpy()

    outer_cv = GroupKFold(n_splits=n_outer_splits)
    fold_r2, fold_mae = [], []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        search = GridSearchCV(
            make_pipeline(),
            PARAM_GRID,
            cv=GroupKFold(n_splits=n_inner_splits),
            scoring="r2",
            n_jobs=-1,
        )
        search.fit(X_train, y_train, groups=groups_train)

        y_pred = search.predict(X_test)
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        fold_r2.append(r2)
        fold_mae.append(mae)
        print(f"  fold {fold}: R²={r2:.3f}  MAE={mae:.3f}  "
              f"best_params={search.best_params_}  n_test={len(test_idx)}")

    print(f"{diagnosis}: mean R²={np.mean(fold_r2):.3f} (±{np.std(fold_r2):.3f})  "
          f"mean MAE={np.mean(fold_mae):.3f}  n={len(y)}  subjects={len(set(groups))}")


def main() -> None:
    for diagnosis in ("CD", "UC"):
        print(f"=== {diagnosis} ===")
        run(diagnosis)
        print()


if __name__ == "__main__":
    main()
