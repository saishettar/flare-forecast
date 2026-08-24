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
