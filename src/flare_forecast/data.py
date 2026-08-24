"""Load HMP2 metadata and taxonomic profiles into aligned model-ready tables.

Activity scores are diagnosis-specific in this cohort (HBI is scored for
Crohn's disease, SCCAI for ulcerative colitis — confirmed empirically:
of metagenomics rows with a non-null HBI, 689/693 are CD; of those with a
non-null SCCAI, 436/452 are UC). There is no single instrument that
applies across both diagnoses in HMP2, so callers pick one
(diagnosis, score_col) pair per model rather than pooling scores across
diagnoses on one shared scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

METADATA_PATH = RAW_DIR / "hmp2_metadata_2018-08-20.csv"
TAXONOMY_PATH = RAW_DIR / "taxonomic_profiles_3.tsv.gz"

# Activity score column, by diagnosis it's actually scored for.
DIAGNOSIS_SCORE_COL = {"CD": "hbi", "UC": "sccai"}


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(METADATA_PATH, low_memory=False)


def load_species_taxonomy() -> pd.DataFrame:
    """Species-level relative abundances, samples x features.

    The merged table stacks every taxonomic rank (kingdom..species) in one
    file with '|'-delimited lineage strings as the index. Using every rank
    at once double-counts signal (a phylum's abundance is the sum of its
    species), so we keep only the deepest rank (species, 6 '|' separators)
    to get a single non-redundant, (near-)compositional feature set.
    """
    tax = pd.read_csv(TAXONOMY_PATH, sep="\t", index_col=0)
    species = tax[tax.index.str.count(r"\|") == 6]
    species = species.rename(index=lambda s: s.rsplit("|", 1)[-1])
    samples = species.T / 100.0  # source values are percentages (sum to 100), not proportions
    samples.index = samples.index.str.removesuffix("_profile")
    samples.index.name = "External ID"
    return samples


def build_baseline_dataset(diagnosis: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Cross-sectional (same-timepoint) X/y/groups for one diagnosis' activity score.

    Returns:
        X: species-level relative abundances, one row per metagenomics sample.
        y: the diagnosis' activity score (hbi for CD, sccai for UC) at that sample.
        groups: Participant ID per row, for subject-grouped cross-validation
            (HMP2 has up to 24 repeated-measures timepoints per subject —
            an ungrouped split leaks the same patient's microbiome across
            train/test folds).
    """
    if diagnosis not in DIAGNOSIS_SCORE_COL:
        raise ValueError(f"diagnosis must be one of {list(DIAGNOSIS_SCORE_COL)}, got {diagnosis!r}")
    score_col = DIAGNOSIS_SCORE_COL[diagnosis]

    meta = load_metadata()
    mgx = meta[(meta["data_type"] == "metagenomics") & (meta["diagnosis"] == diagnosis)]
    mgx = mgx.dropna(subset=[score_col]).set_index("External ID")

    species = load_species_taxonomy()
    joined = mgx[[score_col, "Participant ID"]].join(species, how="inner")

    X = joined.drop(columns=[score_col, "Participant ID"])
    y = joined[score_col]
    groups = joined["Participant ID"]
    return X, y, groups


def _load_scored_mgx(diagnosis: str) -> pd.DataFrame:
    """Metagenomics samples for one diagnosis, with score + week_num, species-joined."""
    if diagnosis not in DIAGNOSIS_SCORE_COL:
        raise ValueError(f"diagnosis must be one of {list(DIAGNOSIS_SCORE_COL)}, got {diagnosis!r}")
    score_col = DIAGNOSIS_SCORE_COL[diagnosis]

    meta = load_metadata()
    mgx = meta[(meta["data_type"] == "metagenomics") & (meta["diagnosis"] == diagnosis)]
    mgx = mgx.dropna(subset=[score_col, "week_num"]).set_index("External ID")

    species = load_species_taxonomy()
    joined = mgx[[score_col, "Participant ID", "week_num"]].join(species, how="inner")
    return joined.rename(columns={score_col: "score"})


def build_forecast_dataset(
    diagnosis: str, min_gap_weeks: float = 2, max_gap_weeks: float = 4
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """(t, t+1) forecasting pairs: microbiome + score at t -> score at t+1.

    For each subject, every same-subject pair of metagenomics timepoints
    (t_i, t_j) with week_j - week_i in [min_gap_weeks, max_gap_weeks] is
    one row -- not just consecutive visits, since HMP2's sampling
    intervals are irregular (median 2 weeks, but ranging 0-19; see
    scripts/eda_phase1.py). [2, 4] weeks was chosen because it covers
    the bulk of naturally occurring gaps (1044/1508 = 69% of consecutive
    HMP2 metagenomics gaps fall in [2,4]) and matches SCOPE's target
    forecast horizon.

    Returns:
        X_t: species-level relative abundances at timepoint t (source).
        score_t: activity score at t -- the naive "persistence" predictor
            (does the microbiome add anything beyond just today's score?).
        y: activity score at t+1 (target, 2-4 weeks after t).
        groups: Participant ID, for subject-grouped/LOSO cross-validation.
        gap: weeks between t and t+1 (diagnostic only, not a feature).
    """
    scored = _load_scored_mgx(diagnosis)
    feature_cols = [c for c in scored.columns if c not in ("score", "Participant ID", "week_num")]

    rows_X, rows_score_t, rows_y, rows_groups, rows_gap = [], [], [], [], []
    for pid, g in scored.groupby("Participant ID"):
        g = g.sort_values("week_num")
        weeks = g["week_num"].to_numpy()
        for i in range(len(g)):
            for j in range(len(g)):
                gap = weeks[j] - weeks[i]
                if min_gap_weeks <= gap <= max_gap_weeks:
                    rows_X.append(g.iloc[i][feature_cols].to_numpy())
                    rows_score_t.append(g.iloc[i]["score"])
                    rows_y.append(g.iloc[j]["score"])
                    rows_groups.append(pid)
                    rows_gap.append(gap)

    X_t = pd.DataFrame(np.vstack(rows_X), columns=feature_cols)
    score_t = pd.Series(rows_score_t, name="score_t")
    y = pd.Series(rows_y, name="score_t1")
    groups = pd.Series(rows_groups, name="Participant ID")
    gap = pd.Series(rows_gap, name="gap_weeks")
    return X_t, score_t, y, groups, gap
