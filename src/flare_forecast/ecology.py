"""Low-dimensional ecological summary features, as an alternative to raw
578-species relative abundance.

The raw-species ElasticNet (see train_forecast.py) zeroes out nearly every
species coefficient under honest LOSO-tuned regularization -- at ~51-83
training subjects, 578 compositional features is too little signal per
parameter for Lasso/ElasticNet to find anything worth keeping. These
functions collapse a sample's species profile to a handful of ecologically
meaningful numbers instead, on the theory that *those* might sit in a more
learnable n/p regime:

- shannon_diversity / species_richness: standard alpha-diversity summaries.
  Loss of diversity is a well-established correlate of IBD activity.
- dysbiosis_score: median Bray-Curtis dissimilarity to a reference cohort
  of non-IBD samples, i.e. "how far is this composition from a typical
  healthy gut" -- the same construct Lloyd-Price et al. (2019) used to
  characterize HMP2 dysbiosis, computed here from scratch against this
  cohort's own non-IBD subjects (429 samples, 27 subjects) since the
  original per-subject dysbiosis scores aren't republished in the raw
  downloads.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def shannon_diversity(X: np.ndarray) -> np.ndarray:
    X = np.clip(X, 1e-12, None)
    mask = X > 1e-12
    p = np.where(mask, X, 1.0)  # avoid log(0); masked entries don't contribute
    return -np.sum(np.where(mask, p * np.log(p), 0.0), axis=1)


def species_richness(X: np.ndarray) -> np.ndarray:
    return (X > 0).sum(axis=1)


def dysbiosis_score(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Median Bray-Curtis dissimilarity from each row of X to `reference`."""
    dists = cdist(X, reference, metric="braycurtis")
    return np.median(dists, axis=1)
