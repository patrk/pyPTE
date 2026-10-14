"""Estimator bias: raw PTE on unrelated signals, and the tools that remove it.

The plug-in entropy estimate is biased by roughly (occupied cells - 1) / 2N,
and PTE combines four of them. Scott's rule, the default, sizes bins for a
one-dimensional histogram, so the three-dimensional joint histogram PTE needs
holds about one sample per cell and independent noise reports around 0.75
bits. The default stays because dPTE needs the phase resolution and the bias
cancels in its ratio; "auto" binning and bias_correction serve raw PTE.
"""

import numpy as np
import pytest

from pyPTE import PTE
from pyPTE.core.pyPTE import get_bincount_auto


def independent(n, seed):
    return np.random.default_rng(seed).standard_normal((2, n))


def mean_raw_pte(n, **kwargs):
    values = [PTE(independent(n, seed), **kwargs)[1][0, 1] for seed in range(3)]
    return float(np.mean(values))


@pytest.mark.parametrize("n", [4000, 64000])
def test_auto_binning_has_small_bias_on_independent_noise(n):
    assert mean_raw_pte(n, binning="auto") < 0.05


def test_bias_correction_removes_what_remains():
    corrected = mean_raw_pte(16000, binning="auto", bias_correction=True)
    assert abs(corrected) < 0.005
    assert abs(corrected) < mean_raw_pte(16000, binning="auto")


def test_default_scott_binning_is_biased_as_documented():
    """If this stops holding, the docstring and README warning need revisiting."""
    assert mean_raw_pte(4000) > 0.3


def test_auto_bincount_targets_samples_per_cell():
    # (n / 20) ** (1/3) cells per axis, never fewer than 3
    assert get_bincount_auto(160_000) == 20
    assert get_bincount_auto(20_000) == 10
    assert get_bincount_auto(100) == 3


def test_identical_channels_give_undefined_dpte():
    """Zero transfer both ways is 0/0, not 0.5 and not rounding noise."""
    rng = np.random.default_rng(0)
    a = np.sin(np.arange(6000) / 4) + 0.3 * rng.standard_normal(6000)
    b = np.roll(a, 7) + 0.3 * rng.standard_normal(6000)
    dPTE, raw = PTE(np.vstack([a, b, a]))
    assert np.isnan(dPTE[0, 2]) and np.isnan(dPTE[2, 0])
    assert raw[0, 2] == 0.0 and raw[2, 0] == 0.0
    assert np.isfinite(dPTE[0, 1])
