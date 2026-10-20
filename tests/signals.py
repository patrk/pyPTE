"""Simulated recordings shared by the tests.

Coupled pairs give both channels the same noise, so the direction comes from
the lag alone. dPTE on broadband data otherwise points from the cleaner channel
to the noisier one, whatever the coupling (#97), and a test whose driver
happens to be the cleaner channel passes for the wrong reason.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 250.0


def oscillator(rng, n, freq=10.0):
    """A 10 Hz rhythm whose phase drifts, as real rhythms do.

    It starts at a random phase. Starting every rhythm at zero would lock two
    unrelated rhythms at a fixed phase offset in every short epoch, which a
    group test across epochs reads as a consistent direction (#99).
    """
    start = rng.uniform(0.0, 2 * np.pi)
    drift = np.cumsum(rng.normal(0.0, 0.4 / np.sqrt(FS), n))
    return np.sin(2 * np.pi * freq * np.arange(n) / FS + drift + start)


def coupled(rng, n, *, driver_noise=0.3, target_noise=0.3, lag=12):
    """Channel 0 drives channel 1 by `lag` samples.

    The lag is cut from a longer rhythm rather than rolled, since a circular
    roll would wrap the end of the driver onto the start of the target.
    """
    rhythm = oscillator(rng, n + lag)
    driver = rhythm[lag:] + driver_noise * rng.standard_normal(n)
    target = rhythm[:n] + target_noise * rng.standard_normal(n)
    return np.vstack([driver, target])


def noisy_driver_pair(n=30000, seed=0, driver_noise=1.0):
    """A noisy channel driving a clean one, so SNR points the wrong way."""
    return coupled(
        np.random.default_rng(seed), n, driver_noise=driver_noise, target_noise=0.1
    )


def mixed_pair(n=30000, seed=0):
    """One source seen instantaneously by two channels with unequal noise."""
    rng = np.random.default_rng(seed)
    source = oscillator(rng, n)
    return np.vstack(
        [
            source + 0.2 * rng.standard_normal(n),
            0.8 * source + 1.0 * rng.standard_normal(n),
        ]
    )


def band_pass(data, low=8.0, high=12.0):
    sos = butter(4, [low, high], btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, data, axis=1)
