"""Flare forecasting: what actually predicts activity 2-4 weeks out.

Predicts activity score 2-4 weeks ahead (score_t1) from timepoint t's gut
microbiome composition, evaluated with leave-one-subject-out CV so a
patient's own repeated-measures samples never appear in both train and
test.

Models per diagnosis, to isolate what the microbiome actually adds:
  1. persistence      = y_pred = score_t (the trivial "nothing changed" guess)
  2. score_regression = LinearRegression(score_t) -> score_t1 (a fitted
                         linear recalibration of persistence, e.g. mean
                         reversion; still no microbiome data)
  3. microbiome        = ElasticNet on species abundance at t only
  4. combined          = ElasticNet on species abundance + score_t
  5. ecology           = ElasticNet on ecology.py's low-dim summary features
                         (diversity/richness/dysbiosis score at t, and their
                         deltas from the prior visit) instead of raw species
  6. ecology_combined  = ecology features + score_t
  7. pathway           = ElasticNet on HUMAnN pathway abundance (476
                         unstratified pathways) instead of species taxonomy
  8. pathway_combined  = pathway features + score_t

(2) exists because comparing (4) only against (1) is misleading. (4) can
beat (1) purely by learning a slope and intercept for score_t, since
persistence forces slope=1, intercept=0, with zero contribution from any
species. That is exactly what happened in an earlier run (see the
SHAP-check commit): (4)'s species coefficients are almost all zero, so
the honest comparison for whether the microbiome adds anything is (4) vs
(2), not (4) vs (1). (5)/(6) exist because raw species may simply be too
high-dimensional (578 features) for 51-83 training subjects to find
signal in even when it exists (see data.py's build_forecast_dataset_
ecology). (7)/(8) exist because taxonomic identity is not the only way
to represent the microbiome: different species can fill the same
metabolic role, so a functional (pathway) view might carry signal a
taxonomic one does not.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The pathway features (476 columns, many correlated since pathways share genes)
# trip ElasticNet's convergence check at the weak-regularization end of the grid
# search regardless of max_iter. The fit still returns a usable solution and
# GridSearchCV picks among all of them by held-out score, so this is noise, not
# a correctness problem, but 50+ warnings per fold makes the real output
# unreadable.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flare_forecast.data import (  # noqa: E402
    build_forecast_dataset,
    build_forecast_dataset_ecology,
    build_forecast_dataset_metabolomics,
    load_pathway_abundance,
)
from flare_forecast.features import ArcsinSqrtTransform, Log1pTransform, PrevalenceFilter  # noqa: E402

PARAM_GRID = {
    "model__alpha": np.logspace(-3, 1, 9),
    "model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
}
N_INNER_SPLITS = 4


def make_compositional_pipeline(n_feature_cols: int, n_total_cols: int, transform) -> Pipeline:
    """ElasticNet pipeline. Feature columns get prevalence-filter + `transform` +
    scale; any extra non-compositional column (score_t) only gets scaled, since
    transform (arcsin-sqrt or log1p) assumes something about the feature
    columns' scale that score_t does not share."""
    feature_pipe = Pipeline([
        ("prevalence", PrevalenceFilter(min_prevalence=0.1)),
        ("transform", transform),
        ("scale", StandardScaler()),
    ])
    transformers = [("features", feature_pipe, list(range(n_feature_cols)))]
    if n_total_cols > n_feature_cols:
        transformers.append(
            ("extra", StandardScaler(), list(range(n_feature_cols, n_total_cols)))
        )
    preprocess = ColumnTransformer(transformers)
    return Pipeline([("preprocess", preprocess), ("model", ElasticNet(max_iter=50_000))])


def make_pipeline(n_species_cols: int, n_total_cols: int) -> Pipeline:
    return make_compositional_pipeline(n_species_cols, n_total_cols, ArcsinSqrtTransform())


def make_pathway_pipeline(n_pathway_cols: int, n_total_cols: int) -> Pipeline:
    return make_compositional_pipeline(n_pathway_cols, n_total_cols, Log1pTransform())


def make_metabolomics_pipeline(n_metabolite_cols: int, n_total_cols: int) -> Pipeline:
    # Metabolite intensities are also skewed and non-negative but not a bounded
    # proportion, same reasoning as make_pathway_pipeline.
    return make_compositional_pipeline(n_metabolite_cols, n_total_cols, Log1pTransform())


def make_ecology_pipeline() -> Pipeline:
    """Ecology summary features are already plain numeric (diversity index,
    a count, a distance, deltas thereof) -- no compositional prevalence
    filter or arcsin-sqrt transform needed, just scale + ElasticNet."""
    return Pipeline([("scale", StandardScaler()), ("model", ElasticNet(max_iter=50_000))])


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def loso_generic(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, pipeline_factory
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
            pipeline_factory(),
            PARAM_GRID,
            cv=GroupKFold(n_splits=n_inner),
            scoring="r2",
            n_jobs=1,
        )
        search.fit(X_train, y_train, groups=groups_train)
        y_pred_all.extend(search.predict(X_test))
        y_true_all.extend(y_test)
    return np.array(y_true_all), np.array(y_pred_all)


def loso_elasticnet(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_species_cols: int
) -> tuple[np.ndarray, np.ndarray]:
    return loso_generic(X, y, groups, lambda: make_pipeline(n_species_cols, X.shape[1]))


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
    X_t, score_t, y, groups, gap, _week_t = build_forecast_dataset(diagnosis)
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

    # 5/6. low-dimensional ecology summary features, alone and + score_t
    X_eco, score_t_eco, y_eco, groups_eco, _ = build_forecast_dataset_ecology(diagnosis)
    X_eco, score_t_eco = X_eco.to_numpy(), score_t_eco.to_numpy()
    y_eco, groups_eco = y_eco.to_numpy(), groups_eco.to_numpy()

    y_true, y_pred = loso_generic(X_eco, y_eco, groups_eco, make_ecology_pipeline)
    record("ecology", y_true, y_pred)

    X_eco_combined = np.hstack([X_eco, score_t_eco.reshape(-1, 1)])
    y_true, y_pred = loso_generic(X_eco_combined, y_eco, groups_eco, make_ecology_pipeline)
    record("ecology_combined", y_true, y_pred)

    # 7/8. HUMAnN pathway abundance (476 unstratified pathways), alone and + score_t
    X_pw, score_t_pw, y_pw, groups_pw, _, _ = build_forecast_dataset(
        diagnosis, feature_loader=load_pathway_abundance
    )
    n_pathway_cols = X_pw.shape[1]
    X_pw, score_t_pw = X_pw.to_numpy(), score_t_pw.to_numpy()
    y_pw, groups_pw = y_pw.to_numpy(), groups_pw.to_numpy()

    y_true, y_pred = loso_generic(
        X_pw, y_pw, groups_pw, lambda: make_pathway_pipeline(n_pathway_cols, X_pw.shape[1])
    )
    record("pathway", y_true, y_pred)

    X_pw_combined = np.hstack([X_pw, score_t_pw.reshape(-1, 1)])
    y_true, y_pred = loso_generic(
        X_pw_combined, y_pw, groups_pw,
        lambda: make_pathway_pipeline(n_pathway_cols, X_pw_combined.shape[1]),
    )
    record("pathway_combined", y_true, y_pred)

    print()
    return results


def run_metabolomics(diagnosis: str) -> dict:
    """Metabolomics as a secondary enrichment pass over the ~30% paired
    subset, not folded into run()'s main comparison. persistence and
    score_regression are recomputed on this same smaller subset rather
    than reused from run()'s full-data numbers, since comparing a
    metabolomics model against a baseline fit on a different, larger set
    of subjects would not be a fair comparison."""
    X_mbx, score_t, y, groups, gap, _week_t = build_forecast_dataset_metabolomics(diagnosis)
    n_metabolite_cols = X_mbx.shape[1]
    X_mbx, score_t = X_mbx.to_numpy(), score_t.to_numpy()
    y, groups = y.to_numpy(), groups.to_numpy()
    n_subjects = len(set(groups))

    print(f"=== {diagnosis} metabolomics subset (n={len(y)} pairs, {n_subjects} subjects) ===")
    results = {"diagnosis": diagnosis, "n_pairs": len(y), "n_subjects": n_subjects, "models": {}}

    def record(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        score_r2, score_mae = r2(y_true, y_pred), float(np.mean(np.abs(y_true - y_pred)))
        results["models"][name] = {"r2": score_r2, "mae": score_mae}
        print(f"  {name:<20}: R2={score_r2:.3f}  MAE={score_mae:.3f}")

    record("persistence (subset)", y, score_t)

    y_true, y_pred = loso_score_regression(score_t, y, groups)
    record("score_regression (subset)", y_true, y_pred)

    y_true, y_pred = loso_generic(
        X_mbx, y, groups,
        lambda: make_metabolomics_pipeline(n_metabolite_cols, X_mbx.shape[1]),
    )
    record("metabolomics", y_true, y_pred)

    X_mbx_combined = np.hstack([X_mbx, score_t.reshape(-1, 1)])
    y_true, y_pred = loso_generic(
        X_mbx_combined, y, groups,
        lambda: make_metabolomics_pipeline(n_metabolite_cols, X_mbx_combined.shape[1]),
    )
    record("metabolomics_combined", y_true, y_pred)

    print()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "forecast.json",
        help="Where to write per-diagnosis LOSO results as JSON (set to '' to skip).",
    )
    parser.add_argument(
        "--metabolomics",
        action="store_true",
        help="Also run the metabolomics enrichment pass (slower, smaller paired subset, "
             "written to results/forecast_metabolomics.json).",
    )
    args = parser.parse_args()

    all_results = [run(diagnosis) for diagnosis in ("CD", "UC")]

    if args.save_json and str(args.save_json):
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(all_results, indent=2))
        print(f"wrote {args.save_json}")

    if args.metabolomics:
        mbx_results = [run_metabolomics(diagnosis) for diagnosis in ("CD", "UC")]
        mbx_path = Path(__file__).resolve().parent.parent / "results" / "forecast_metabolomics.json"
        mbx_path.write_text(json.dumps(mbx_results, indent=2))
        print(f"wrote {mbx_path}")


if __name__ == "__main__":
    main()
