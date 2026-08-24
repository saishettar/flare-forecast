"""Biological plausibility check for the "combined" model, via SHAP.

An earlier run found the "combined" model (microbiome + current score)
clears the persistence baseline, but a modest R^2 improvement on ~700
samples does not rule out the model leaning on noise. This is a
plausibility check independent of the accuracy number: fit the same
combined ElasticNet on all data (single grouped hyperparameter search,
not LOSO, since this is about inspecting one model's learned
coefficients, not re-proving that earlier validation), then check
whether the species it weights most heavily match taxa already
implicated in IBD by prior literature.

Reference direction (Lloyd-Price et al. 2019, *Nature*, the HMP2 source
paper) for what "the model found something real" would look like:
  - Overrepresented in dysbiosis/flares: Proteobacteria bloom, especially
    Escherichia coli and other Enterobacteriaceae; Ruminococcus gnavus.
  - Depleted in dysbiosis (loss of short-chain-fatty-acid/butyrate
    producers): Faecalibacterium prausnitzii, Roseburia hominis,
    Roseburia intestinalis, Eubacterium rectale, Coprococcus comes,
    Alistipes putredinis, Dialister invisus.

This is a coarse, non-exhaustive reference list for a sanity check, not
a formal enrichment test. Absence from it does not mean a species is
biologically irrelevant, and presence does not prove the model learned
the mechanism the literature describes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import shap
from sklearn.model_selection import GridSearchCV, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flare_forecast.data import build_forecast_dataset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_forecast import PARAM_GRID, make_pipeline  # noqa: E402

FLARE_ASSOCIATED = {
    "Escherichia_coli", "Klebsiella_pneumoniae", "Enterobacter_cloacae",
    "Ruminococcus_gnavus", "Clostridium_difficile", "Fusobacterium_nucleatum",
}
PROTECTIVE_BUTYRATE_PRODUCERS = {
    "Faecalibacterium_prausnitzii", "Roseburia_hominis", "Roseburia_intestinalis",
    "Eubacterium_rectale", "Coprococcus_comes", "Alistipes_putredinis",
    "Dialister_invisus", "Roseburia_inulinivorans",
}


def species_short_name(lineage_or_name: str) -> str:
    return lineage_or_name.rsplit("s__", 1)[-1] if "s__" in lineage_or_name else lineage_or_name


def run(diagnosis: str, top_n: int = 20) -> None:
    X_t, score_t, y, groups, gap, _week_t = build_forecast_dataset(diagnosis)
    feature_names = list(X_t.columns)
    n_species_cols = X_t.shape[1]

    X = np.hstack([X_t.to_numpy(), score_t.to_numpy().reshape(-1, 1)])
    y, groups = y.to_numpy(), groups.to_numpy()

    search = GridSearchCV(
        make_pipeline(n_species_cols, X.shape[1]),
        PARAM_GRID,
        cv=GroupKFold(n_splits=5),
        scoring="r2",
        n_jobs=1,
    )
    search.fit(X, y, groups=groups)
    pipeline = search.best_estimator_
    print(f"=== {diagnosis}: best_params={search.best_params_} "
          f"(in-sample R2={search.score(X, y):.3f}, not a held-out estimate) ===")

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    keep_mask = preprocess.named_transformers_["species"].named_steps["prevalence"].keep_mask_
    kept_species = [name for name, keep in zip(feature_names, keep_mask) if keep]
    all_feature_names = kept_species + ["score_t"]

    n_nonzero_species = int(np.count_nonzero(model.coef_[:-1]))
    print(f"nonzero species coefficients: {n_nonzero_species}/{len(kept_species)} "
          f"(score_t coef={model.coef_[-1]:+.3f}) -- if this is ~0, the model is "
          "leaning on score_t, not the microbiome")

    X_transformed = preprocess.transform(X)
    masker = shap.maskers.Independent(X_transformed, max_samples=X_transformed.shape[0])
    explainer = shap.LinearExplainer(model, masker)
    shap_values = explainer(X_transformed).values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_signed_shap = shap_values.mean(axis=0)

    order = np.argsort(-mean_abs_shap)[:top_n]
    print(f"{'feature':<45} {'mean|SHAP|':>10} {'mean SHAP':>10}  literature match")
    n_flare_match, n_protective_match = 0, 0
    for i in order:
        name = all_feature_names[i]
        short = species_short_name(name)
        tag = ""
        if short in FLARE_ASSOCIATED:
            tag = "flare-associated (overrepresented in IBD)"
            n_flare_match += 1
        elif short in PROTECTIVE_BUTYRATE_PRODUCERS:
            tag = "protective (butyrate producer, depleted in IBD)"
            n_protective_match += 1
        print(f"{name:<45} {mean_abs_shap[i]:>10.4f} {mean_signed_shap[i]:>+10.4f}  {tag}")

    print(f"-> {n_flare_match + n_protective_match}/{top_n} top-SHAP features match the "
          f"reference IBD taxa list ({n_flare_match} flare-associated, "
          f"{n_protective_match} protective)\n")


def main() -> None:
    for diagnosis in ("CD", "UC"):
        run(diagnosis)


if __name__ == "__main__":
    main()
