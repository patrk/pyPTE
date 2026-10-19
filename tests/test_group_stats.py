import numpy as np
import pytest

from pyPTE import PTE, cluster_permutation, group_contrast, group_test

FS = 250.0


def oscillator(rng, n, freq=10.0):
    drift = np.cumsum(rng.normal(0.0, 0.4 / np.sqrt(FS), n))
    return np.sin(2 * np.pi * freq * np.arange(n) / FS + drift)


def epoch(n, coupling, seed):
    """One short epoch; channel 0 drives channel 1 when coupling > 0."""
    rng = np.random.default_rng(seed)
    driver = oscillator(rng, n) + 0.3 * rng.standard_normal(n)
    target = (
        coupling * np.roll(driver, 12)
        + max(1.0 - coupling, 0.0) * oscillator(rng, n)
        + 0.3 * rng.standard_normal(n)
    )
    return np.vstack([driver, target])


def matrices(n_epochs, n_samples, coupling, offset=0):
    return np.array(
        [PTE(epoch(n_samples, coupling, s + offset))[0] for s in range(n_epochs)]
    )


# --------------------------------------------------------------------------
# group_test
# --------------------------------------------------------------------------


def test_short_epochs_beat_a_single_long_recording():
    """The reason this exists: consistency across epochs is very powerful.

    Eighty epochs of a third of a second - 26 seconds of data in total -
    resolve a coupling that a single continuous minute leaves marginal under
    the surrogate test.
    """
    result = group_test(matrices(80, 80, 0.9))

    assert result.significant[0, 1]
    assert result.p_values[0, 1] < 1e-6
    assert result.effect[0, 1] > 0.5


def test_independent_epochs_are_not_significant():
    result = group_test(matrices(80, 80, 0.0))

    assert not result.significant.any()
    assert result.p_values[0, 1] > 0.05


def test_ttest_agrees_with_wilcoxon_on_a_clear_effect():
    data = matrices(40, 250, 0.9)

    wilcoxon = group_test(data, method="wilcoxon")
    ttest = group_test(data, method="ttest")

    assert wilcoxon.significant[0, 1] and ttest.significant[0, 1]
    np.testing.assert_allclose(wilcoxon.effect, ttest.effect)


def test_reference_can_be_moved_for_raw_pte():
    """Raw PTE is tested against 0, not 0.5."""
    raw = np.array([PTE(epoch(250, 0.9, s))[1] for s in range(20)])
    result = group_test(raw, reference=0.1)

    assert result.significant[0, 1]
    assert result.effect[0, 1] > 0


def test_diagonal_is_never_significant():
    result = group_test(matrices(20, 250, 0.9))

    assert not np.diag(result.significant).any()
    np.testing.assert_array_equal(np.diag(result.p_values), 1.0)


def test_group_test_rejects_bad_input():
    with pytest.raises(ValueError, match="n_observations"):
        group_test(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="at least 2"):
        group_test(np.zeros((1, 3, 3)))


# --------------------------------------------------------------------------
# group_contrast
# --------------------------------------------------------------------------


def test_contrast_detects_a_condition_difference():
    coupled = matrices(40, 250, 0.9)
    uncoupled = matrices(40, 250, 0.0)

    result = group_contrast(coupled, uncoupled)

    assert result.significant[0, 1]
    assert result.effect[0, 1] > 0, "condition A should show more A -> B flow"


def test_contrast_finds_nothing_between_equivalent_conditions():
    """Two samples of the same process must not differ."""
    result = group_contrast(matrices(40, 250, 0.9), matrices(40, 250, 0.9, offset=500))
    assert not result.significant.any()


def test_contrast_is_antisymmetric_in_its_arguments():
    a = matrices(30, 250, 0.9)
    b = matrices(30, 250, 0.0)

    forward = group_contrast(a, b)
    reverse = group_contrast(b, a)

    np.testing.assert_allclose(forward.effect, -reverse.effect)
    np.testing.assert_allclose(forward.p_values, reverse.p_values)


def test_contrast_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="matching shapes"):
        group_contrast(np.zeros((5, 3, 3)), np.zeros((4, 3, 3)))
    with pytest.raises(ValueError, match="at least 2"):
        group_contrast(np.zeros((1, 3, 3)), np.zeros((1, 3, 3)))


def test_summary_reports_the_method_and_count():
    result = group_contrast(matrices(20, 250, 0.9), matrices(20, 250, 0.0))
    assert "paired wilcoxon" in result.summary()
    assert "20 observations" in result.summary()


# --------------------------------------------------------------------------
# cluster_permutation
# --------------------------------------------------------------------------


def chain_epoch(n, coupling, seed):
    """a -> b -> c -> d, so the true edges form one connected component."""
    rng = np.random.default_rng(seed)
    signals = [oscillator(rng, n) + 0.3 * rng.standard_normal(n)]
    for _ in range(3):
        signals.append(
            coupling * np.roll(signals[-1], 10)
            + max(1.0 - coupling, 0.0) * oscillator(rng, n)
            + 0.3 * rng.standard_normal(n)
        )
    return np.vstack(signals)


def chain_matrices(n_epochs, n_samples, coupling, offset=0):
    return np.array(
        [PTE(chain_epoch(n_samples, coupling, s + offset))[0] for s in range(n_epochs)]
    )


def test_cluster_finds_a_connected_chain():
    result = cluster_permutation(
        chain_matrices(40, 250, 0.9), n_permutations=200, seed=0
    )

    assert result.clusters, "a coupled chain should form at least one cluster"
    assert result.cluster_p.min() < 0.05
    assert result.significant.any()


def test_cluster_is_calibrated_under_the_null():
    """Independent signals should rarely produce a surviving cluster."""
    hits = 0
    for k in range(8):
        data = chain_matrices(25, 250, 0.0, offset=k * 1000)
        result = cluster_permutation(data, n_permutations=200, seed=k)
        hits += bool(result.significant.any())

    assert hits <= 2, f"false-positive rate looks miscalibrated: {hits}/8"


def test_cluster_detects_a_paired_condition_difference():
    result = cluster_permutation(
        chain_matrices(30, 250, 0.9),
        chain_matrices(30, 250, 0.0),
        n_permutations=200,
        seed=0,
    )
    assert result.significant.any()
    assert result.cluster_p.min() < 0.05


def test_cluster_p_values_are_bounded_away_from_zero():
    n_permutations = 100
    result = cluster_permutation(
        chain_matrices(30, 250, 0.9), n_permutations=n_permutations, seed=0
    )
    assert (result.cluster_p >= 1.0 / (n_permutations + 1)).all()


def test_stricter_threshold_yields_smaller_clusters():
    data = chain_matrices(40, 250, 0.9)

    lenient = cluster_permutation(data, threshold_p=0.05, n_permutations=100, seed=0)
    strict = cluster_permutation(data, threshold_p=0.001, n_permutations=100, seed=0)

    assert strict.threshold > lenient.threshold
    assert strict.significant.sum() <= lenient.significant.sum()


def test_cluster_reports_nothing_when_no_edge_passes_threshold():
    flat = np.full((10, 4, 4), 0.5)
    result = cluster_permutation(flat, n_permutations=50, seed=0)

    assert result.clusters == []
    assert not result.significant.any()
    assert "no edges passed" in result.summary()


def test_cluster_rejects_bad_input():
    with pytest.raises(ValueError, match="n_observations"):
        cluster_permutation(np.zeros((4, 4)), n_permutations=10)
    with pytest.raises(ValueError, match="at least 2"):
        cluster_permutation(np.zeros((1, 3, 3)), n_permutations=10)
    with pytest.raises(ValueError, match="matching shapes"):
        cluster_permutation(np.zeros((5, 3, 3)), np.zeros((4, 3, 3)), n_permutations=10)


def test_cluster_result_n_significant_clusters_respects_alpha():
    from pyPTE.stats import ClusterResult

    res_default = ClusterResult(
        statistic=np.zeros((3, 3)),
        clusters=[np.ones((3, 3), dtype=bool), np.ones((3, 3), dtype=bool)],
        cluster_p=np.array([0.03, 0.08]),
        significant=np.zeros((3, 3), dtype=bool),
        threshold=2.0,
        n_permutations=100,
    )
    assert res_default.alpha == 0.05
    assert res_default.n_significant_clusters == 1

    res_custom = ClusterResult(
        statistic=np.zeros((3, 3)),
        clusters=[np.ones((3, 3), dtype=bool), np.ones((3, 3), dtype=bool)],
        cluster_p=np.array([0.03, 0.08]),
        significant=np.zeros((3, 3), dtype=bool),
        threshold=2.0,
        n_permutations=100,
        alpha=0.10,
    )
    assert res_custom.alpha == 0.10
    assert res_custom.n_significant_clusters == 2

    res_strict = ClusterResult(
        statistic=np.zeros((3, 3)),
        clusters=[np.ones((3, 3), dtype=bool), np.ones((3, 3), dtype=bool)],
        cluster_p=np.array([0.03, 0.08]),
        significant=np.zeros((3, 3), dtype=bool),
        threshold=2.0,
        n_permutations=100,
        alpha=0.01,
    )
    assert res_strict.alpha == 0.01
    assert res_strict.n_significant_clusters == 0


def test_cluster_permutation_stores_alpha_on_result():
    flat = np.full((10, 4, 4), 0.5)
    empty_res = cluster_permutation(flat, n_permutations=50, alpha=0.01, seed=0)
    assert empty_res.alpha == 0.01
    assert empty_res.n_significant_clusters == 0

    data = chain_matrices(30, 250, 0.9)
    res = cluster_permutation(data, n_permutations=100, alpha=0.10, seed=0)
    assert res.alpha == 0.10
    assert res.n_significant_clusters == int((res.cluster_p <= 0.10).sum())


# --------------------------------------------------------------------------
# each pair once (#63, #64)
# --------------------------------------------------------------------------


def antisymmetric_samples(n_obs=12, m=4, seed=0):
    """dPTE-shaped matrices: [j, i] == 1 - [i, j], zero diagonal."""
    rng = np.random.default_rng(seed)
    upper = rng.uniform(0.3, 0.8, (n_obs, m, m))
    samples = np.triu(upper, 1) + np.tril(1.0 - upper.transpose(0, 2, 1), -1)
    samples[:, np.arange(m), np.arange(m)] = 0.0
    return samples


@pytest.mark.parametrize("method", ["wilcoxon", "ttest"])
def test_mirrored_pairs_match_testing_every_pair(method):
    from scipy import stats

    samples = antisymmetric_samples()
    result = group_test(samples, method=method)
    for i, j in zip(*np.nonzero(~np.eye(4, dtype=bool)), strict=True):
        column = samples[:, i, j] - 0.5
        if method == "wilcoxon":
            expected = stats.wilcoxon(column)
        else:
            expected = stats.ttest_1samp(column, 0.0)
        assert result.statistic[i, j] == pytest.approx(expected.statistic)
        assert result.p_values[i, j] == pytest.approx(expected.pvalue)


def test_raw_pte_pairs_are_tested_in_both_orientations():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.0, 0.2, (12, 3, 3))
    raw[:, 0, 1] += 0.1  # 0 -> 1 only; raw PTE has no mirror relation
    result = group_test(raw, reference=0.1)
    assert result.p_values[0, 1] != result.p_values[1, 0]


def test_clusters_hold_each_dpte_pair_once():
    result = cluster_permutation(
        chain_matrices(40, 250, 0.9), n_permutations=200, seed=0
    )
    assert result.clusters
    covered = np.sum(result.clusters, axis=0)
    # no pair appears twice, neither within nor across clusters, nor mirrored
    assert covered.max() == 1
    assert not np.any(covered.astype(bool) & covered.astype(bool).T)
    # clusters are oriented along the flow, a -> b -> c -> d
    assert all((result.statistic[mask] > 0).all() for mask in result.clusters)
    assert np.array_equal(result.significant, result.significant.T)


def test_small_samples_use_every_sign_flip_exactly():
    data = chain_matrices(5, 250, 0.9)
    first = cluster_permutation(data, threshold_p=0.05, n_permutations=1000, seed=0)
    second = cluster_permutation(data, threshold_p=0.05, n_permutations=1000, seed=1)

    assert first.clusters
    assert first.n_permutations == 2**5
    # the observed labelling is one of the 32, so p is a multiple of 1/32
    np.testing.assert_allclose(first.cluster_p * 32, np.round(first.cluster_p * 32))
    assert first.cluster_p.min() >= 1 / 32
    # nothing random is left, so the seed cannot matter
    np.testing.assert_array_equal(first.cluster_p, second.cluster_p)
