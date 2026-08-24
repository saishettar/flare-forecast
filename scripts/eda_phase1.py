"""Phase 1 EDA: feature/sample dimensions and the single- vs multi-omic call.

SCOPE.md flags the core technical risk up front: ~132 subjects but
thousands of taxa/pathway features (classic small-n/high-dimension), and
asks that the single- vs multi-omic decision be resolved here, once real
counts are in hand. This script prints those counts against the raw
downloads in data/raw/ (run scripts/download_data.py first).

Findings (2026-08-24, bioBakery 3.0 / 2018-08-20 metadata release):
    - 130 subjects with >=1 metagenomics sample, 1638 MGX samples total.
    - Feature counts dwarf sample count at every level: 932 taxa,
      22,113 pathways, 167,854 ECs vs. 130 subjects / 1638 samples.
      -> regularization (ElasticNet/Lasso) or a curated/dimension-reduced
      feature set is mandatory, not optional, for any of these.
    - Metabolomics only has 546 samples / 106 subjects, and only 473 of
      the 1638 MGX samples (29%) have a same-timepoint paired MBX sample.
      Fusing metabolomics in from the start would cut the modeling set
      by ~70% before regularization even gets a chance to help.
    - Decision: start single-omic (metagenomics: taxonomy + pathways).
      Revisit metabolomics fusion in Phase 4 as a secondary/enrichment
      pass over the smaller paired subset, not the primary model.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def count_gzipped_rows(path: Path) -> int:
    with gzip.open(path, "rt") as f:
        return sum(1 for _ in f) - 1  # exclude header


def main() -> None:
    meta = pd.read_csv(RAW_DIR / "hmp2_metadata_2018-08-20.csv", low_memory=False)

    mgx = meta[meta["data_type"] == "metagenomics"]
    mbx = meta[meta["data_type"] == "metabolomics"]

    print("=== Sample / subject counts ===")
    print(f"metagenomics: {mgx.shape[0]} samples, {mgx['Participant ID'].nunique()} subjects")
    print(f"metabolomics: {mbx.shape[0]} samples, {mbx['Participant ID'].nunique()} subjects")
    print(f"diagnosis breakdown:\n{meta['diagnosis'].value_counts()}")

    print("\n=== Clinical activity score coverage ===")
    for col in ("hbi", "sccai", "Tube B:Fecal Calprotectin"):
        print(f"{col}: {meta[col].notna().sum()} non-null rows")

    print("\n=== Paired MGX/MBX timepoints (same subject, same week_num) ===")
    mgx_keys = set(zip(mgx["Participant ID"], mgx["week_num"]))
    mbx_keys = set(zip(mbx["Participant ID"], mbx["week_num"]))
    paired = mgx_keys & mbx_keys
    print(f"paired timepoints: {len(paired)} / {len(mgx_keys)} MGX timepoints "
          f"({len(paired) / len(mgx_keys):.0%})")
    print(f"subjects with >=1 paired timepoint: {len(set(s for s, _ in paired))}")

    print("\n=== Feature dimensionality vs. sample count ===")
    tax = pd.read_csv(RAW_DIR / "taxonomic_profiles_3.tsv.gz", sep="\t", index_col=0)
    print(f"taxonomy: {tax.shape[0]} features x {tax.shape[1]} samples")

    n_pathways = count_gzipped_rows(RAW_DIR / "pathabundances_3.tsv.gz")
    print(f"pathways: {n_pathways} features")

    n_ecs = count_gzipped_rows(RAW_DIR / "ecs_3.tsv.gz")
    print(f"ECs: {n_ecs} features")

    print(f"\nfeatures >> subjects at every level ({tax.shape[0]}-{n_ecs} features vs. "
          f"{mgx['Participant ID'].nunique()} subjects) -> regularization/dimensionality "
          "reduction is required, single-omic (metagenomics) is the Phase 1-3 modeling basis.")


if __name__ == "__main__":
    main()
