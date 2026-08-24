import numpy as np

from flare_forecast.ecology import dysbiosis_score, shannon_diversity, species_richness


def test_shannon_diversity_zero_for_single_species():
    X = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(shannon_diversity(X), [0.0])


def test_shannon_diversity_maximal_for_uniform_distribution():
    n = 4
    X = np.array([[1.0 / n] * n])
    np.testing.assert_allclose(shannon_diversity(X), [np.log(n)])


def test_shannon_diversity_ignores_zero_entries():
    # padding a distribution with zero-abundance species should not change its entropy
    X = np.array([[0.5, 0.5, 0.0, 0.0]])
    np.testing.assert_allclose(shannon_diversity(X), [np.log(2)])


def test_species_richness_counts_nonzero_entries():
    X = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    np.testing.assert_array_equal(species_richness(X), [2, 0, 3])


def test_dysbiosis_score_zero_for_identical_reference():
    X = np.array([[0.3, 0.7]])
    reference = np.array([[0.3, 0.7], [0.3, 0.7]])
    np.testing.assert_allclose(dysbiosis_score(X, reference), [0.0])


def test_dysbiosis_score_one_for_disjoint_support():
    X = np.array([[1.0, 0.0]])
    reference = np.array([[0.0, 1.0]])
    np.testing.assert_allclose(dysbiosis_score(X, reference), [1.0])


def test_dysbiosis_score_is_the_median_across_reference_samples():
    X = np.array([[1.0, 0.0]])
    # Bray-Curtis distances to these three reference rows are 0.0, 0.5, 1.0 -- median is 0.5
    reference = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    np.testing.assert_allclose(dysbiosis_score(X, reference), [0.5])
