"""Load HMP2 metadata and taxonomic profiles into aligned model-ready tables.

Activity scores are diagnosis-specific in this cohort (HBI is scored for
Crohn's disease, SCCAI for ulcerative colitis, confirmed empirically:
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

from flare_forecast.ecology import dysbiosis_score, shannon_diversity, species_richness

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

METADATA_PATH = RAW_DIR / "hmp2_metadata_2018-08-20.csv"
TAXONOMY_PATH = RAW_DIR / "taxonomic_profiles_3.tsv.gz"
PATHWAY_PATH = RAW_DIR / "pathabundances_3.tsv.gz"

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

    # 11/1638 samples have exactly 0 abundance across all 578 species -- failed
    # taxonomic profiling runs that still got a row in the merged table, not real
    # biological zero-diversity samples. Left in, they're a division-by-zero trap
    # for Bray-Curtis dysbiosis scoring and a spurious "zero diversity" data point
    # for everything else. Drop them here so every downstream consumer is clean.
    samples = samples[samples.sum(axis=1) > 0]
    return samples


def load_pathway_abundance() -> pd.DataFrame:
    """HUMAnN community-level pathway abundance (copies per million), samples x features.

    Like the taxonomy file, this stacks every stratification level
    (community total, then per-contributing-species breakdowns) in one
    file, with '|'-delimited row names for the stratified rows. Keeping
    only the unstratified rows (no '|') gives one non-redundant feature
    per pathway, 476 of them, after also dropping UNMAPPED/UNINTEGRATED
    (reads that could not be assigned to any pathway at all -- not a
    biological signal about a specific pathway).

    Unlike species relative abundance, these are not proportions in
    [0, 1] (a sample's pathways can and do overlap in which reads they
    draw on), so they get a log1p transform downstream instead of
    arcsin-sqrt.
    """
    path = pd.read_csv(PATHWAY_PATH, sep="\t", index_col=0)
    unstratified = path[~path.index.str.contains(r"\|")]
    unstratified = unstratified[~unstratified.index.isin(["UNMAPPED", "UNINTEGRATED"])]
    samples = unstratified.T
    samples.index = samples.index.str.removesuffix("_pathabundance_cpm")
    samples.index.name = "External ID"

    # Same failed-profiling-run issue as load_species_taxonomy, 14/1638 samples
    # here rather than 11 (mostly, not entirely, the same samples -- a few
    # profile taxonomically but fail functional profiling, or vice versa).
    samples = samples[samples.sum(axis=1) > 0]
    return samples


def build_baseline_dataset(diagnosis: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Cross-sectional (same-timepoint) X/y/groups for one diagnosis' activity score.

    Returns:
        X: species-level relative abundances, one row per metagenomics sample.
        y: the diagnosis' activity score (hbi for CD, sccai for UC) at that sample.
        groups: Participant ID per row, for subject-grouped cross-validation
            (HMP2 has up to 24 repeated-measures timepoints per subject,
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


def load_nonibd_reference_species() -> pd.DataFrame:
    """Species-level relative abundance for the non-IBD reference cohort.

    Used as the "healthy" comparison set for dysbiosis_score -- 429
    metagenomics samples from 27 non-IBD subjects, disjoint from the
    CD/UC subjects being modeled, so using this as a fixed reference in
    any CD/UC subject's LOSO fold introduces no leakage.
    """
    meta = load_metadata()
    ref = meta[(meta["data_type"] == "metagenomics") & (meta["diagnosis"] == "nonIBD")]
    ref = ref.set_index("External ID")
    species = load_species_taxonomy()
    return ref[[]].join(species, how="inner")


def _load_scored_mgx(diagnosis: str, feature_loader=load_species_taxonomy) -> pd.DataFrame:
    """Metagenomics samples for one diagnosis, with score + week_num, feature-joined.

    feature_loader is any zero-arg callable returning a samples x features
    DataFrame indexed by External ID (load_species_taxonomy or
    load_pathway_abundance), so the same join/pairing logic below works
    for either feature representation without duplicating it.
    """
    if diagnosis not in DIAGNOSIS_SCORE_COL:
        raise ValueError(f"diagnosis must be one of {list(DIAGNOSIS_SCORE_COL)}, got {diagnosis!r}")
    score_col = DIAGNOSIS_SCORE_COL[diagnosis]

    meta = load_metadata()
    mgx = meta[(meta["data_type"] == "metagenomics") & (meta["diagnosis"] == diagnosis)]
    mgx = mgx.dropna(subset=[score_col, "week_num"]).set_index("External ID")

    features = feature_loader()
    joined = mgx[[score_col, "Participant ID", "week_num"]].join(features, how="inner")
    return joined.rename(columns={score_col: "score"})


def build_forecast_dataset(
    diagnosis: str,
    min_gap_weeks: float = 2,
    max_gap_weeks: float = 4,
    feature_loader=load_species_taxonomy,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """(t, t+1) forecasting pairs: microbiome + score at t -> score at t+1.

    For each subject, every same-subject pair of metagenomics timepoints
    (t_i, t_j) with week_j - week_i in [min_gap_weeks, max_gap_weeks] is
    one row -- not just consecutive visits, since HMP2's sampling
    intervals are irregular (median 2 weeks, but ranging 0-19; see
    scripts/eda.py). [2, 4] weeks was chosen because it covers
    the bulk of naturally occurring gaps (1044/1508 = 69% of consecutive
    HMP2 metagenomics gaps fall in [2,4]) and matches the project's
    target forecast horizon.

    feature_loader picks the feature representation: species-level
    taxonomy by default, or load_pathway_abundance for HUMAnN pathways.

    Returns:
        X_t: features at timepoint t (source).
        score_t: activity score at t -- the naive "persistence" predictor
            (does the microbiome add anything beyond just today's score?).
        y: activity score at t+1 (target, 2-4 weeks after t).
        groups: Participant ID, for subject-grouped/LOSO cross-validation.
        gap: weeks between t and t+1 (diagnostic only, not a feature).
        week_t: week_num of the source timepoint (diagnostic/plotting only,
            not a feature -- lets a caller reconstruct each pair's real
            position on a subject's timeline via week_t and week_t + gap).
    """
    scored = _load_scored_mgx(diagnosis, feature_loader)
    feature_cols = [c for c in scored.columns if c not in ("score", "Participant ID", "week_num")]

    rows_X, rows_score_t, rows_y, rows_groups, rows_gap, rows_week_t = [], [], [], [], [], []
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
                    rows_week_t.append(weeks[i])

    X_t = pd.DataFrame(np.vstack(rows_X), columns=feature_cols)
    score_t = pd.Series(rows_score_t, name="score_t")
    y = pd.Series(rows_y, name="score_t1")
    groups = pd.Series(rows_groups, name="Participant ID")
    gap = pd.Series(rows_gap, name="gap_weeks")
    week_t = pd.Series(rows_week_t, name="week_t")
    return X_t, score_t, y, groups, gap, week_t


ECOLOGY_FEATURE_COLS = [
    "shannon_t", "richness_t", "dysbiosis_t",
    "delta_shannon", "delta_richness", "delta_dysbiosis", "gap_to_prev", "has_prior",
]


def build_forecast_dataset_ecology(
    diagnosis: str, min_gap_weeks: float = 2, max_gap_weeks: float = 4
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """(t, t+1) forecasting pairs, but with low-dimensional ecological summary
    features instead of raw 578-species abundance -- see ecology.py for why.

    Adds trajectory features (delta_* / gap_to_prev / has_prior) computed
    against each subject's immediately preceding metagenomics visit
    (regardless of that visit's gap to t; 91-92% of CD/UC timepoints have
    one). When no prior visit exists, deltas are 0 and has_prior=0 flags it
    rather than silently treating "no data" as "no change".

    Returns: X_ecology, score_t, y, groups, gap -- same shapes/meaning as
    build_forecast_dataset, X_ecology has columns ECOLOGY_FEATURE_COLS.
    """
    scored = _load_scored_mgx(diagnosis)
    species_cols = [c for c in scored.columns if c not in ("score", "Participant ID", "week_num")]
    species_matrix = scored[species_cols].to_numpy()

    reference = load_nonibd_reference_species()[species_cols].to_numpy()
    scored = scored.assign(
        shannon=shannon_diversity(species_matrix),
        richness=species_richness(species_matrix),
        dysbiosis=dysbiosis_score(species_matrix, reference),
    )

    rows_X, rows_score_t, rows_y, rows_groups, rows_gap = [], [], [], [], []
    for pid, g in scored.groupby("Participant ID"):
        g = g.sort_values("week_num").reset_index(drop=True)
        weeks = g["week_num"].to_numpy()
        for i in range(len(g)):
            for j in range(len(g)):
                gap = weeks[j] - weeks[i]
                if not (min_gap_weeks <= gap <= max_gap_weeks):
                    continue
                if i > 0:
                    prev = g.iloc[i - 1]
                    delta_shannon = g.iloc[i]["shannon"] - prev["shannon"]
                    delta_richness = g.iloc[i]["richness"] - prev["richness"]
                    delta_dysbiosis = g.iloc[i]["dysbiosis"] - prev["dysbiosis"]
                    gap_to_prev = weeks[i] - prev["week_num"]
                    has_prior = 1
                else:
                    delta_shannon = delta_richness = delta_dysbiosis = 0.0
                    gap_to_prev = 0.0
                    has_prior = 0
                rows_X.append([
                    g.iloc[i]["shannon"], g.iloc[i]["richness"], g.iloc[i]["dysbiosis"],
                    delta_shannon, delta_richness, delta_dysbiosis, gap_to_prev, has_prior,
                ])
                rows_score_t.append(g.iloc[i]["score"])
                rows_y.append(g.iloc[j]["score"])
                rows_groups.append(pid)
                rows_gap.append(gap)

    X = pd.DataFrame(rows_X, columns=ECOLOGY_FEATURE_COLS)
    score_t = pd.Series(rows_score_t, name="score_t")
    y = pd.Series(rows_y, name="score_t1")
    groups = pd.Series(rows_groups, name="Participant ID")
    gap = pd.Series(rows_gap, name="gap_weeks")
    return X, score_t, y, groups, gap
