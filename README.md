# pyPTE: Phase Transfer Entropy in Python

[![PyPI](https://img.shields.io/pypi/v/pyPTE.svg)](https://pypi.org/project/pyPTE/)
[![Python](https://img.shields.io/pypi/pyversions/pyPTE.svg)](https://pypi.org/project/pyPTE/)
[![CI](https://github.com/patrk/pyPTE/actions/workflows/main.yml/badge.svg)](https://github.com/patrk/pyPTE/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

`pyPTE` estimates **directed connectivity** between oscillatory signals: given
two channels of EEG, MEG or any rhythmic time series, it answers *which one is
leading the other*.

It is worth being precise about the claim, because it is narrower than it first
appears:

> **pyPTE recovers the direction of information flow between a pair of signals.
> It does not recover the wiring diagram of a network.**

Direction is reliable. Which connections are *direct* is not recoverable from a
bivariate measure, and the section on [what can fool it](#what-can-fool-it)
shows exactly how that fails, with numbers. Every claim below is produced by a
script in [`examples/`](examples/) that asserts its own result, so if something
here stops being true, the test suite says so.

Based on:

- Lobier et al., 2014: [Phase transfer entropy: A novel phase-based measure for directed connectivity in networks coupled by oscillatory interactions](http://dx.doi.org/10.1016/j.neuroimage.2013.08.056)
- Hillebrand et al., 2016: [Direction of information flow in large-scale resting-state networks is frequency-dependent](http://dx.doi.org/10.1073/pnas.1515657113)

---

## Installation

```bash
pip install pyPTE          # or: uv add pyPTE
```

The core needs only NumPy and SciPy. Adapters are optional:

```bash
pip install "pyPTE[mne]"      # MNE-Python adapter (also installs pandas)
pip install "pyPTE[pandas]"   # pandas adapter
```

Requires Python 3.11 or newer. Dependencies are deliberately unpinned at the top
end so `pyPTE` resolves alongside whatever NumPy and SciPy the rest of your
environment needs.

## Quickstart

`PTE` takes an `(n_channels, n_samples)` array and returns two matrices, where
entry `[i, j]` describes flow from channel `i` to channel `j`.

```python
import numpy as np
from pyPTE import PTE

rng = np.random.default_rng(0)
t = np.arange(8000) / 250.0

driver = np.sin(2 * np.pi * 10 * t) + 0.2 * rng.standard_normal(t.size)
target = 0.9 * np.roll(driver, 12) + 0.4 * rng.standard_normal(t.size)

dPTE, raw_PTE = PTE(np.vstack([driver, target]))

print(dPTE[0, 1])  # ~0.80 -> driver leads target
print(dPTE[1, 0])  # ~0.20 -> the reverse is suppressed
```

`raw_PTE` is transfer entropy in bits. `dPTE` is the direction-normalised form,
where `dPTE[i, j] + dPTE[j, i] == 1`: above `0.5` means net flow from `i` to
`j`, and exactly `0.5` means no preferred direction.

**A raw `dPTE` value on its own is not a finding.** See
[running a real analysis](#running-a-real-analysis).

## How it works

For each ordered channel pair, `PTE` does five things:

1. **Extract phase.** A Hilbert transform turns each channel into an analytic
   signal; the angle between its real and imaginary parts is the instantaneous
   phase, on `[-pi, pi)`.
2. **Estimate a delay.** The analysis lag is derived from the mean number of
   samples between zero crossings, so it adapts to the dominant rhythm.
3. **Bin the phase.** Scott's rule sets the bin width from the data, which keeps
   the histogram roughly as well-populated at 100 samples as at 100,000.
4. **Count joint states.** Occurrences of `(Y_future, Y_now, X_now)` are counted
   and turned into entropies.
5. **Combine.** `PTE = H(Ypr,Y) + H(Y,X) - H(Y) - H(Ypr,Y,X)`, which is the
   conditional mutual information `I(Y_future ; X | Y)` — how much better you
   can predict `Y`'s next phase knowing `X`, beyond knowing `Y`'s own past.

Equivalently, in the usual notation:

$$PTE_{X \to Y} = H(Y_{t+1} \mid Y_t) - H(Y_{t+1} \mid Y_t, X_t)$$

Because it is a conditional mutual information, raw PTE is non-negative and
bounded above by `log2(n_bins)`. Both are asserted in the test suite.

Filter to a band of interest **before** calling `PTE`. Phase is only meaningful
for a reasonably narrowband signal, and the measure says nothing about which
frequency an interaction lives at — that comes from how you filter.

### The two free choices, and why they are parameters

Steps 2 and 3 are not determined by the method, and both change the answer:

```python
PTE(signal, binning="scott", delay="zero-crossing")  # the defaults
```

**`binning`** accepts `"scott"` (default), `"hillebrand"`, or an integer bin
count. Worth knowing: Hillebrand et al. 2016 specify
`exp(0.626 + 0.4·ln(N_s − δ − 1))`, which is **3–4× more bins than Scott's rule**
— 68 against 20 at 8,000 samples. pyPTE has always used Scott's rule, and that
is kept as the default deliberately: at 68 bins the three-way histogram has
314,000 cells for 8,000 samples and is hopelessly sparse, whereas Scott's rule
holds roughly one sample per cell across a wide range of data lengths.

It is not a cosmetic difference. On coupled signals Scott's rule reports
`dPTE ≈ 0.68` where Hillebrand's reports `≈ 0.56`. Both agree on direction, and
both sit at 0.5 when there is nothing to find, but the effect sizes are not
comparable. Pass `binning="hillebrand"` to match the paper.

**`delay`** accepts `"zero-crossing"` (default, and what Hillebrand specifies),
`"phase-increment"`, or an integer. The two rules estimate the same quantity and
agree within a sample on narrowband signals; they diverge on broadband or
DC-offset signals, where neither is trustworthy. A signal containing several
rhythms gives a delay tracking the fastest one, which is another reason to
filter first.

## Running a real analysis

Reporting raw `dPTE` values is the single most common way to get a wrong answer
out of this library. Two independent signals that merely differ in
signal-to-noise ratio produce `dPTE` around `0.73` — indistinguishable from
genuine coupling. The value has to be tested against a null.

Which test depends on the shape of your data.

### One long continuous recording

Compare the observed matrix against **time-shifted surrogates**: each channel is
circularly shifted by an independent random offset, which preserves every
per-channel property exactly while destroying cross-channel timing. Any apparent
direction that survives comes from the interaction rather than the channels.

```python
from pyPTE import surrogate_test

result = surrogate_test(signal, n_surrogates=200, seed=0)
print(result.summary())
result.significant  # boolean mask, FDR corrected
result.p_values
```

### Many short epochs, trials or subjects — usually better

This is what M/EEG data normally looks like, and it is **far more sensitive**,
because it asks whether an effect is *consistent* rather than whether one number
is extreme. Twenty-five one-second epochs settle questions that sixty continuous
seconds leave marginal.

```python
import numpy as np
from pyPTE import PTE, group_test, group_contrast

matrices = np.array([PTE(epoch)[0] for epoch in epochs])  # (n_epochs, m, m)

group_test(matrices)  # is connectivity above chance?
group_contrast(task, rest)  # does it differ between conditions?
```

A single epoch may be far too short for a reliable estimate on its own; what
matters is that the estimate is unbiased, so noise averages out across
observations. Wilcoxon signed-rank is the default, matching the convention in
the M/EEG literature.

### Correcting for multiple comparisons

An `m`-channel recording produces `m * (m - 1)` ordered pairs — 9,900 for 100
channels. Two corrections are provided, answering different questions:

| | bounds | best when | claim |
|---|---|---|---|
| `benjamini_hochberg` (used by default) | share of false edges among those reported | effects are isolated | per edge |
| `cluster_permutation` | chance of reporting any spurious component | effects span connected edges | per component |

```python
from pyPTE import cluster_permutation

clusters = cluster_permutation(task, rest, n_permutations=1000, seed=0)
```

`cluster_permutation` is the network-based-statistic form of cluster-based
permutation testing, which is what M/EEG reviewers usually expect. Its claim is
weaker than it looks: a significant cluster means *the component contains an
effect*, not that every edge in it is real.

Bonferroni is deliberately not offered. At 9,900 tests it retains no power.

### The recommended pipeline

```
band-pass filter  ->  epoch  ->  PTE per epoch  ->  group test
                                              ->  FDR or cluster correction
                                              ->  interpret only what survives
```

## What can fool it

Each of these is demonstrated by a script that asserts the number, not merely
described.

### Unequal signal-to-noise ratio fabricates direction

Two **completely independent** oscillators differing only in noise level:

| noise A | noise B | dPTE[A→B] |
|---:|---:|---:|
| 0.2 | 0.2 | 0.492 |
| 0.2 | 0.4 | 0.684 |
| 0.2 | 0.8 | **0.797** |

The noisier channel is harder to predict from its own past, so the cleaner one
looks like a driver. Real recordings always differ in SNR between channels.

A surrogate test removes it cleanly: for such a pair the null mean lands at
`0.739` against an observed `0.736`, and it is correctly rejected — while
genuine coupling survives at `p = 0.005`. In one worked case the artefact even
has a **higher** raw `dPTE` (0.719) than the real coupling (0.664), so ranking
pairs by raw value picks the wrong one.

→ [`two_node_coupling.py`](examples/two_node_coupling.py),
[`significance_testing.py`](examples/significance_testing.py)

### Indirect paths outrank direct ones

In a chain `a → b → c → d` where only the one-hop links exist:

| separation | mean dPTE | real edge? |
|---|---:|---|
| 1 hop | 0.582 | **yes** |
| 2 hops | 0.634 | no |
| 3 hops | **0.702** | no |

`dPTE` rises monotonically with path length, because a longer path accumulates
more phase lag. **Thresholding by dPTE returns close to the inverse of the true
network.** This is the fundamental limit of any bivariate measure, and more data
makes it worse: in a neural-mass simulation, recall saturated at 60 s while
precision fell from 0.50 to 0.36 by 200 s as indirect paths became detectable.

There is therefore an optimum recording length for network reconstruction, and
it is not "as much as possible".

→ [`epoched_analysis.py`](examples/epoched_analysis.py),
[`neural_mass_network.py`](examples/neural_mass_network.py)

### Symmetric coupling is invisible — by design

Strongly but *symmetrically* coupled channels sit at exactly `0.500`. That is
the correct answer, not a miss: `dPTE` measures **net** direction, and a
balanced bidirectional link has none. Score it against raw connectivity and you
will penalise it for being right.

→ [`kuramoto_network.py`](examples/kuramoto_network.py)

### On a real connectome, structural recovery largely fails

Against The Virtual Brain's directed 76-region tract-tracing connectome, pyPTE
separates one-way edges from unconnected pairs at **AUC 0.62** — above chance,
but far from the 0.96–1.00 the same estimator reaches on sparse synthetic
networks. At 27% density nearly every region pair is joined by a short indirect
path, which is the previous caveat operating at whole-brain scale.

Scale bites twice. With `m` regions there are `m * (m - 1)` pairs — 5,700 here —
so the strictest FDR threshold falls below `1e-5`, while `N` surrogates cannot
produce a p-value below `1 / (N + 1)`. Edgewise surrogate testing at this scale
needs thousands of surrogates; `cluster_permutation` is the practical
alternative.

→ [`tvb_connectome.py`](examples/tvb_connectome.py)

### Detection is model-dependent

There is no universal sample requirement. Phase oscillators resolve in a few
thousand samples; realistic Jansen-Rit columns needed 60 s or more, and were
strongly sensitive to the transmission delay. Effect size depends on the
dynamics, the SNR and the lag, which is why group-level testing is the robust
answer rather than a rule of thumb.

### Other things to know

- **Very strong coupling reduces detectability.** Oscillators synchronise and
  their phases become redundant, so dPTE peaks at intermediate coupling.
- **Common drivers** create apparent links: if `a` drives both `b` and `c`, then
  `b` and `c` will look connected.
- **Volume conduction** in sensor-space M/EEG produces zero-lag mixing that no
  phase-based measure can separate from genuine coupling. Prefer source space.
- **Cost is `O(m^2)`** in channels. A 100-channel montage over 8,000 samples
  takes about 0.8 s; 100 surrogates on it takes about 1.4 minutes.

## Examples

Every example builds a system with known ground truth, runs `pyPTE`, and
**asserts** the result. If an assertion fails, the claim it makes is no longer
true.

```bash
uv sync --group examples
uv run python -m examples.two_node_coupling      # add --quick for a faster run
```

| example | what it establishes |
|---|---|
| [`two_node_coupling`](examples/two_node_coupling.py) | direction is recovered; SNR asymmetry fakes it |
| [`kuramoto_network`](examples/kuramoto_network.py) | known directed graphs recovered, AUC 0.96–1.00 |
| [`neural_mass_network`](examples/neural_mass_network.py) | recording length vs precision, on Jansen-Rit columns |
| [`significance_testing`](examples/significance_testing.py) | what a surrogate test does, drawn out |
| [`epoched_analysis`](examples/epoched_analysis.py) | full epoched two-condition pipeline |
| [`tvb_connectome`](examples/tvb_connectome.py) | a real directed connectome, where recovery mostly fails |

See [`examples/README.md`](examples/README.md) for details.

## API

```python
from pyPTE import (
    PTE,  # dPTE and raw PTE from an (m, n) array
    surrogate_test,  # significance for one continuous recording
    group_test,  # across epochs/trials/subjects, vs chance
    group_contrast,  # paired comparison of two conditions
    cluster_permutation,  # network-based statistic correction
    benjamini_hochberg,  # FDR, on any array of p-values
)
```

Adapters, which return labelled `pandas.DataFrame` results:

```python
from pyPTE.adapters.mne_adapter import PTE_from_mne, interpolate_mne
from pyPTE.adapters.pandas_adapter import PTE_from_dataframe
```

## Related work and acknowledgements

The example models here are self-contained on purpose, so they run in CI without
extra dependencies. If you want richer or more citable ground-truth signals,
these are the simulators worth knowing about:

- **[The Virtual Brain](https://github.com/the-virtual-brain/tvb-root)**
  (`tvb-library`, GPL-3.0-or-later) — the reference whole-brain platform, and
  the one worth reaching for first. It ships a genuinely **directed** 76-region
  connectome from tract-tracing rather than a symmetric DTI matrix, which is
  what a directional measure needs as ground truth, plus ~30 population models
  and EEG/MEG forward monitors. Sanz Leon et al., *Front. Neuroinform.* 7:10
  (2013).
  → **[`examples/tvb_connectome.py`](examples/tvb_connectome.py)** runs pyPTE
  against that connectome, and is the most informative example here because the
  result is largely negative.
- **[PyRates](https://github.com/pyrates-neuroscience/PyRates)** (GPL-3.0) — a
  much smaller dependency footprint, with explicit per-edge directed coupling
  and delays. No bundled connectome, so you supply the topology, which is what
  the self-contained models here already do; the gain over them is citability
  rather than capability. Gast et al., *PLOS ONE* 14(12):e0225900 (2019).
- [neurolib](https://github.com/neurolib-dev/neurolib) (MIT) — whole-brain
  modelling with several neural mass models. **No example here**, deliberately:
  its bundled HCP connectome is symmetric, so it carries no net direction for a
  directional measure to recover, and the project has had little activity since
  December 2024.
- [brainmass](https://github.com/chaobrain/brainmass) (Apache-2.0), part of the
  [BrainX](https://brainx.chaobrain.com/) ecosystem — differentiable neural mass
  models on JAX. **No example here**, deliberately: at v0.1.1 the API is still
  moving, and it pulls in the whole JAX stack for a demonstration the models
  above already provide.

`pyPTE` depends on none of them.

## Contributing

Contributions are welcome — issues, suggestions and pull requests alike.

```bash
git clone https://github.com/patrk/pyPTE && cd pyPTE
uv sync --all-extras
uv run pytest
```

Before opening a PR: `uv run ruff check .`, `uv run ruff format .`,
`uv run mypy pyPTE tests`.

## License

`pyPTE` is released under GPL-3.0-or-later. See [LICENSE](LICENSE).
