"""Phase 3 — the real headline: flare forecasting.

Predicts activity score 2-4 weeks ahead (score_t1) from timepoint t's gut
microbiome composition, evaluated with **leave-one-subject-out** CV (the
validation shape named in SCOPE.md's target resume headline) so a
patient's own repeated-measures samples never appear in both train and
test.

Four models per diagnosis, to isolate what the microbiome actually adds:
  1. persistence      — y_pred = score_t (the trivial "nothing changed" guess)
  2. score_regression — LinearRegression(score_t) -> score_t1 (a *fitted*
                         linear recalibration of persistence, e.g. mean
                         reversion; still no microbiome data)
  3. microbiome        — ElasticNet on species abundance at t only
  4. combined          — ElasticNet on species abundance + score_t

(2) exists because comparing (4) only against (1) is misleading: (4) can
beat (1) purely by *learning a slope/intercept* for score_t (persistence
forces slope=1, intercept=0), with zero contribution from any species.
That's exactly what happened here (see Phase 4 SHAP check) -- (4)'s
species coefficients are ~all zero, so the honest comparison for "does
microbiome add anything" is (4) vs (2), not (4) vs (1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flare_forecast.data import build_forecast_dataset  # noqa: E402
from flare_forecast.features import ArcsinSqrtTransform, PrevalenceFilter  # noqa: E402

PARAM_GRID = {
    "model__alpha": np.logspace(-3, 1, 9),
    "model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
}
N_INNER_SPLITS = 4


def make_pipeline(n_species_cols: int, n_total_cols: int) -> Pipeline:
    """ElasticNet pipeline. species columns get prevalence-filter + arcsin-sqrt +
    scale; any extra non-compositional column (score_t) only gets scaled --
    arcsin-sqrt assumes a [0,1] proportion, which score_t is not."""
    species_pipe = Pipeline([
        ("prevalence", PrevalenceFilter(min_prevalence=0.1)),
        ("arcsin_sqrt", ArcsinSqrtTransform()),
        ("scale", StandardScaler()),
    ])
    transformers = [("species", species_pipe, list(range(n_species_cols)))]
    if n_total_cols > n_species_cols:
        transformers.append(
            ("extra", StandardScaler(), list(range(n_species_cols, n_total_cols)))
        )
    preprocess = ColumnTransformer(transformers)
    return Pipeline([("preprocess", preprocess), ("model", ElasticNet(max_iter=10_000))])


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def loso_elasticnet(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_species_cols: int
) -> tuple[np.ndarray, np.ndarray]:
    """LOSO predictions for an ElasticNet pipeline, alpha/l1_ratio tuned per fold."""
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        n_inner = min(N_INNER_SPLITS, len(set(groups_train)))
        search = GridSearchCV(
            make_pipeline(n_species_cols, X.shape[1]),
            PARAM_GRID,
            cv=GroupKFold(n_splits=n_inner),
            scoring="r2",
            n_jobs=1,
        )
        search.fit(X_train, y_train, groups=groups_train)
        y_pred_all.extend(search.predict(X_test))
        y_true_all.extend(y_test)
    return np.array(y_true_all), np.array(y_pred_all)


def loso_score_regression(
    score_t: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """LOSO predictions for a plain LinearRegression(score_t) -> score_t1."""
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in logo.split(score_t.reshape(-1, 1), y, groups):
        model = LinearRegression()
        model.fit(score_t[train_idx].reshape(-1, 1), y[train_idx])
        y_pred_all.extend(model.predict(score_t[test_idx].reshape(-1, 1)))
        y_true_all.extend(y[test_idx])
    return np.array(y_true_all), np.array(y_pred_all)


def run(diagnosis: str) -> dict:
    X_t, score_t, y, groups, gap = build_forecast_dataset(diagnosis)
    n_species_cols = X_t.shape[1]
    X_t, score_t, y, groups = X_t.to_numpy(), score_t.to_numpy(), y.to_numpy(), groups.to_numpy()
    n_subjects = len(set(groups))

    print(f"=== {diagnosis} (n={len(y)} pairs, {n_subjects} subjects) ===")
    results = {"diagnosis": diagnosis, "n_pairs": len(y), "n_subjects": n_subjects, "models": {}}

    def record(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        score_r2, score_mae = r2(y_true, y_pred), float(np.mean(np.abs(y_true - y_pred)))
        results["models"][name] = {"r2": score_r2, "mae": score_mae}
        print(f"  {name:<11}: R2={score_r2:.3f}  MAE={score_mae:.3f}")

    # 1. persistence: y_pred = score_t, no fitting needed.
    record("persistence", y, score_t)

    # 2. fitted linear recalibration of score_t alone (no microbiome)
    y_true, y_pred = loso_score_regression(score_t, y, groups)
    record("score_regression", y_true, y_pred)

    # 3. microbiome only
    y_true, y_pred = loso_elasticnet(X_t, y, groups, n_species_cols)
    record("microbiome", y_true, y_pred)

    # 4. microbiome + score_t
    X_combined = np.hstack([X_t, score_t.reshape(-1, 1)])
    y_true, y_pred = loso_elasticnet(X_combined, y, groups, n_species_cols)
    record("combined", y_true, y_pred)

    print()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "phase3_forecast.json",
        help="Where to write per-diagnosis LOSO results as JSON (set to '' to skip).",
    )
    args = parser.parse_args()

    all_results = [run(diagnosis) for diagnosis in ("CD", "UC")]

    if args.save_json and str(args.save_json):
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(all_results, indent=2))
        print(f"wrote {args.save_json}")


if __name__ == "__main__":
    main()
