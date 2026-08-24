"""Preprocessing for microbiome and pathway abundance features.

Three sklearn-compatible transformers, meant to sit inside a Pipeline so
they are refit per-CV-fold (on train data only) rather than leaking
test-fold statistics into feature selection/scaling:

- PrevalenceFilter: most of the 578 species are near-universally absent
  (a typical gut microbiome sample carries a few hundred of them). Rare
  species contribute mostly noise to a 130-subject regression, so we
  drop any species present in fewer than `min_prevalence` of training
  samples before modeling. Also useful for the pathway table, where 47%
  of entries are zero.
- ArcsinSqrtTransform: the standard variance-stabilizing transform for
  relative-abundance (proportion) data, used throughout the original
  HMP2 analysis (Lloyd-Price et al. 2019). Relative abundances are
  heavily right-skewed and bounded in [0, 1]; arcsin(sqrt(x)) spreads
  out the near-zero values a log transform would otherwise compress.
- Log1pTransform: for features that are skewed and non-negative but not
  bounded proportions, like HUMAnN pathway abundance in copies per
  million. Arcsin-sqrt assumes a [0, 1] proportion, which these are not.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PrevalenceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_prevalence: float = 0.1):
        self.min_prevalence = min_prevalence

    def fit(self, X, y=None):
        X = np.asarray(X)
        prevalence = (X > 0).mean(axis=0)
        self.keep_mask_ = prevalence >= self.min_prevalence
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.keep_mask_]


class ArcsinSqrtTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.arcsin(np.sqrt(np.clip(X, 0.0, 1.0)))


class Log1pTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.log1p(np.clip(X, 0.0, None))
