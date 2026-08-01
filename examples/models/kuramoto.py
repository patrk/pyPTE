"""Kuramoto phase oscillators with directed, delayed coupling.

The classic Kuramoto model couples oscillators symmetrically and instantly,
which produces synchrony but no directed information flow. Making the coupling
matrix asymmetric and giving it a transmission delay creates a genuine sender
and receiver per edge, which is what a directed connectivity measure should be
able to recover.
"""

import numpy as np
import numpy.typing as npt


def simulate(
    coupling: npt.NDArray,
    *,
    n_samples: int,
    fs: float = 250.0,
    delay: float = 0.012,
    freq: float = 10.0,
    freq_spread: float = 0.5,
    noise: float = 0.3,
    burn_in: float = 2.0,
    seed: int = 0,
) -> npt.NDArray:
    """Integrate delayed Kuramoto oscillators and return their observable signal.

    ``coupling[i, j]`` is the strength with which oscillator ``i`` drives
    oscillator ``j``, so a non-zero entry means information should flow from
    ``i`` to ``j``.

    Parameters
    ----------
    coupling : numpy.ndarray
        m x m directed coupling matrix; the diagonal is ignored
    n_samples : int
        number of samples to return, after the burn-in is discarded
    fs : float
        sampling rate in Hz
    delay : float
        transmission delay in seconds, the lag that makes coupling directional
    freq : float
        centre frequency of the oscillators in Hz
    freq_spread : float
        standard deviation of the per-oscillator natural frequency in Hz
    noise : float
        phase diffusion strength, in radians per sqrt(second)
    burn_in : float
        seconds of transient discarded before returning samples

    Returns
    -------
    signal : numpy.ndarray
        m x n_samples array of sin(phase), the oscillators' observable
    """
    coupling = np.asarray(coupling, dtype=float)
    m = coupling.shape[0]
    if coupling.shape != (m, m):
        raise ValueError(f"coupling must be square, got {coupling.shape}")

    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    delay_steps = max(int(round(delay * fs)), 1)
    n_burn = int(round(burn_in * fs))
    n_total = n_burn + n_samples

    omega = 2 * np.pi * rng.normal(freq, freq_spread, m)

    # incoming[i, j] is the drive from j into i, the transpose of the argument
    incoming = coupling.T.copy()
    np.fill_diagonal(incoming, 0.0)

    # ring buffer of exactly delay_steps entries, so reading a slot before
    # overwriting it yields the phase from exactly delay_steps ago
    history = rng.uniform(0, 2 * np.pi, size=(delay_steps, m))
    theta = history[-1].copy()

    out = np.empty((n_total, m))
    sqrt_dt = np.sqrt(dt)

    for step in range(n_total):
        slot = step % delay_steps
        delayed = history[slot]
        # phase difference of every delayed sender against every current receiver
        interaction = np.sin(delayed[np.newaxis, :] - theta[:, np.newaxis])
        drive = np.sum(incoming * interaction, axis=1)

        theta = theta + (omega + drive) * dt + noise * sqrt_dt * rng.standard_normal(m)
        history[slot] = theta
        out[step] = theta

    return np.sin(out[n_burn:]).T


def ring_with_shortcut(m: int, strength: float = 6.0) -> npt.NDArray:
    """A directed ring, i -> i+1, so ground truth is unambiguous."""
    coupling = np.zeros((m, m))
    for i in range(m):
        coupling[i, (i + 1) % m] = strength
    return coupling


def global_coupling(m: int, strength: float = 6.0) -> npt.NDArray:
    """Mean-field coupling: every oscillator drives every other equally.

    The classic Kuramoto arrangement, normalised by m so the total drive per
    oscillator does not grow with network size. It is symmetric, so it carries
    no net direction anywhere and a directional measure should report nothing.
    Included precisely because that is a useful negative control.
    """
    coupling = np.full((m, m), strength / m)
    np.fill_diagonal(coupling, 0.0)
    return coupling


def local_coupling(
    m: int, strength: float = 6.0, neighbours: int = 1, directed: bool = True
) -> npt.NDArray:
    """Nearest-neighbour coupling on a ring, out to `neighbours` steps.

    With ``directed=True`` each oscillator drives only its clockwise
    neighbours, producing travelling activity with a definite direction. With
    ``directed=False`` the coupling is symmetric, which synchronises locally
    but again leaves no net direction to find.

    Contrasting local with global coupling is the distinction the original
    examples drew: mean-field influence spread thinly over the whole network,
    versus strong influence confined to a neighbourhood.
    """
    coupling = np.zeros((m, m))
    for i in range(m):
        for step in range(1, neighbours + 1):
            coupling[i, (i + step) % m] = strength
            if not directed:
                coupling[i, (i - step) % m] = strength
    return coupling


def two_groups(
    group_size: int = 4,
    within: float = 2.0,
    across: float = 3.0,
    n_bridges: int = 3,
) -> npt.NDArray:
    """Two symmetrically wired groups joined by one-way bridges from A to B.

    Coupling inside each group is deliberately symmetric, so those pairs are
    strongly coupled but carry no net direction. A directional measure should
    therefore report nothing for them, which makes this topology a test of
    whether pyPTE distinguishes direction from mere coupling.
    """
    m = 2 * group_size
    coupling = np.zeros((m, m))

    for group_start in (0, group_size):
        idx = range(group_start, group_start + group_size)
        for i in idx:
            for j in idx:
                if i != j:
                    coupling[i, j] = within

    for k in range(min(n_bridges, group_size)):
        coupling[k, group_size + k] = across
    return coupling


def net_flow(coupling: npt.NDArray) -> npt.NDArray:
    """Ground truth for a directional measure: where does flow actually favour i -> j?

    A symmetric pair cancels, so only asymmetric coupling counts as a directed
    edge. Comparing dPTE against raw connectivity instead would penalise it for
    correctly reporting no direction on a bidirectional link.
    """
    coupling = np.asarray(coupling, dtype=float)
    return coupling > coupling.T
