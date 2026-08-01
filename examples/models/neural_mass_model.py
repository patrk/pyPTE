"""Coupled Jansen-Rit neural mass models.

Each node is a cortical column described by three interacting populations -
pyramidal cells, excitatory interneurons and inhibitory interneurons - whose
post-synaptic potentials produce an alpha-band rhythm for the standard
parameters. Coupling several columns with a transmission delay gives a network
whose directed connectivity is known by construction, but whose signals are far
more physiologically plausible than phase oscillators.

The state per node is six-dimensional: three post-synaptic potentials and their
derivatives. The observable, standing in for an EEG channel, is the net
potential at the pyramidal population, y1 - y2.

Reference
---------
Jansen & Rit (1995), Biological Cybernetics 73:357-366.
"""

import numpy as np
import numpy.typing as npt

# standard parameter set producing alpha activity
A = 3.25  # excitatory synaptic gain (mV)
B = 22.0  # inhibitory synaptic gain (mV)
A_RATE = 100.0  # excitatory time constant, 1/s
B_RATE = 50.0  # inhibitory time constant, 1/s
E0 = 2.5  # half of the maximum firing rate (1/s)
V0 = 6.0  # potential at half maximum firing (mV)
R = 0.56  # sigmoid steepness (1/mV)
C = 135.0  # global connectivity within a column


def sigmoid(v: npt.NDArray) -> npt.NDArray:
    """Population firing rate as a function of membrane potential."""
    return 2.0 * E0 / (1.0 + np.exp(R * (V0 - v)))


def simulate(
    coupling: npt.NDArray,
    *,
    n_samples: int,
    fs: float = 250.0,
    dt: float = 1.0 / 2000.0,
    delay: float = 0.012,
    input_mean: float = 220.0,
    input_sd: float = 22.0,
    burn_in: float = 2.0,
    seed: int = 0,
) -> npt.NDArray:
    """Integrate a network of Jansen-Rit columns and return their EEG-like output.

    ``coupling[i, j]`` is the strength with which column ``i`` drives column
    ``j``, so a non-zero entry means information should flow from ``i`` to
    ``j``. Drive enters the target's excitatory population, delayed, as the
    source's firing rate.

    Integration uses stochastic Heun rather than Euler-Maruyama: the
    deterministic part of this system is stiff enough at a = 100 that a
    first-order step visibly distorts the waveform.

    Parameters
    ----------
    coupling : numpy.ndarray
        m x m directed coupling matrix; the diagonal is ignored
    n_samples : int
        number of samples returned, at `fs`, after burn-in is discarded
    fs : float
        output sampling rate in Hz; the integrator runs at 1/dt and downsamples
    dt : float
        integration step in seconds, which must stay well below the 10 ms
        membrane time constant
    delay : float
        inter-column transmission delay in seconds
    input_mean, input_sd : float
        mean and standard deviation of the background pulse density driving
        each column, the stochastic input of the original model
    burn_in : float
        seconds of transient discarded before returning samples

    Returns
    -------
    signal : numpy.ndarray
        m x n_samples array of y1 - y2 per column, in mV
    """
    coupling = np.asarray(coupling, dtype=float)
    m = coupling.shape[0]
    if coupling.shape != (m, m):
        raise ValueError(f"coupling must be square, got {coupling.shape}")

    decimation = int(round(1.0 / (fs * dt)))
    if decimation < 1:
        raise ValueError("fs must be lower than the integration rate 1/dt")

    rng = np.random.default_rng(seed)
    delay_steps = max(int(round(delay / dt)), 1)
    n_burn = int(round(burn_in / dt))
    n_steps = n_burn + n_samples * decimation

    # incoming[i, j] is drive from j into i, the transpose of the argument
    incoming = coupling.T.copy()
    np.fill_diagonal(incoming, 0.0)

    # y[:, 0:3] are post-synaptic potentials, y[:, 3:6] their derivatives
    y = np.zeros((m, 6))
    y[:, :3] = rng.normal(0.0, 0.5, size=(m, 3))

    history = np.zeros((delay_steps, m))

    def drift(
        state: npt.NDArray, delayed_output: npt.NDArray, pulse_density: npt.NDArray
    ) -> npt.NDArray:
        y0, y1, y2, y3, y4, y5 = state.T
        pyramidal = y1 - y2

        network_drive = incoming @ sigmoid(delayed_output)

        d = np.empty_like(state)
        d[:, 0] = y3
        d[:, 1] = y4
        d[:, 2] = y5
        d[:, 3] = A * A_RATE * sigmoid(pyramidal) - 2 * A_RATE * y3 - A_RATE**2 * y0
        d[:, 4] = (
            A * A_RATE * (pulse_density + 0.8 * C * sigmoid(C * y0) + network_drive)
            - 2 * A_RATE * y4
            - A_RATE**2 * y1
        )
        d[:, 5] = (
            B * B_RATE * 0.25 * C * sigmoid(0.25 * C * y0)
            - 2 * B_RATE * y5
            - B_RATE**2 * y2
        )
        return d

    out = np.empty((n_steps, m))
    for step in range(n_steps):
        slot = step % delay_steps
        delayed_output = history[slot]

        # the background input is redrawn every step and enters the drift, as
        # in the original model. Scaling it as a Wiener increment instead would
        # make the noise swamp the limit cycle that produces the alpha rhythm.
        pulse_density = rng.normal(input_mean, input_sd, size=m)

        # Heun: predict with Euler, then average the two slopes
        k1 = drift(y, delayed_output, pulse_density)
        predictor = y + k1 * dt
        k2 = drift(predictor, delayed_output, pulse_density)
        y = y + 0.5 * (k1 + k2) * dt

        pyramidal = y[:, 1] - y[:, 2]
        history[slot] = pyramidal
        out[step] = pyramidal

    return out[n_burn::decimation][:n_samples].T
