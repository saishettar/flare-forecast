import numpy as np

from flare_forecast.features import ArcsinSqrtTransform, Log1pTransform, PrevalenceFilter


def test_prevalence_filter_keeps_columns_at_or_above_threshold():
    # col 0: present in 3/4 rows (0.75); col 1: present in 1/4 (0.25); col 2: present in 0/4 (0.0)
    X = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    f = PrevalenceFilter(min_prevalence=0.5)
    f.fit(X)
    np.testing.assert_array_equal(f.keep_mask_, [True, False, False])
    np.testing.assert_array_equal(f.transform(X), X[:, [0]])


def test_prevalence_filter_applies_train_mask_to_new_data():
    X_train = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    f = PrevalenceFilter(min_prevalence=0.5).fit(X_train)
    X_test = np.array([[0.5, 9.0]])  # col 1 nonzero here, but fit already decided to drop it
    np.testing.assert_array_equal(f.transform(X_test), [[0.5]])


def test_arcsin_sqrt_known_values():
    X = np.array([[0.0, 1.0, 0.25]])
    out = ArcsinSqrtTransform().fit_transform(X)
    expected = np.array([[0.0, np.pi / 2, np.arcsin(0.5)]])
    np.testing.assert_allclose(out, expected)


def test_arcsin_sqrt_clips_out_of_range_proportions():
    # relative abundances should be in [0, 1], but guard against float slop just past the bounds
    X = np.array([[-0.001, 1.001]])
    out = ArcsinSqrtTransform().fit_transform(X)
    np.testing.assert_allclose(out, [[0.0, np.pi / 2]])


def test_log1p_known_values():
    X = np.array([[0.0, np.e - 1]])
    out = Log1pTransform().fit_transform(X)
    np.testing.assert_allclose(out, [[0.0, 1.0]])


def test_log1p_clips_negative_values():
    X = np.array([[-5.0, 0.0]])
    out = Log1pTransform().fit_transform(X)
    np.testing.assert_allclose(out, [[0.0, 0.0]])
