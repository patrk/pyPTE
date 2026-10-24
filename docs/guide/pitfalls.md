# What can fool it

Each of these is demonstrated by a script that asserts the number, not merely
described.

## Unequal signal-to-noise ratio fabricates direction

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

→ [`two_node_coupling.py`](https://github.com/patrk/pyPTE/blob/main/examples/two_node_coupling.py),
[`significance_testing.py`](https://github.com/patrk/pyPTE/blob/main/examples/significance_testing.py)

## Indirect paths outrank direct ones

In a chain `a → b → c → d` where only the one-hop links exist, with equal
noise on every channel, band-passed, and time-reversal-differenced:

| separation | mean dPTE | real edge? |
|---|---:|---|
| 1 hop | 0.576 | **yes** |
| 2 hops | 0.625 | no |
| 3 hops | **0.659** | no |

`dPTE` rises monotonically with path length, because a longer path accumulates
more phase lag, and PTE detects a longer lag more easily. **Thresholding by
dPTE returns close to the inverse of the true network.** If noise also builds
up along the chain, as it does when each node adds its own, the effect is
stronger still, but that part is the signal-to-noise artefact, and time
reversal removes it.

The lag effect is the fundamental limit of any bivariate measure, and more data
makes it worse: in a neural-mass simulation, recall saturated at 30 s while
precision fell from 0.50 to 0.25 by 160 s as indirect paths became detectable.

There is therefore an optimum recording length for network reconstruction, and
it is not "as much as possible".

→ [`epoched_analysis.py`](https://github.com/patrk/pyPTE/blob/main/examples/epoched_analysis.py),
[`neural_mass_network.py`](https://github.com/patrk/pyPTE/blob/main/examples/neural_mass_network.py)

## Symmetric coupling is invisible — by design

Strongly but *symmetrically* coupled channels sit at exactly `0.500`. That is
the correct answer, not a miss: `dPTE` measures **net** direction, and a
balanced bidirectional link has none. Score it against raw connectivity and you
will penalise it for being right.

→ [`kuramoto_network.py`](https://github.com/patrk/pyPTE/blob/main/examples/kuramoto_network.py)

## On a real connectome, structural recovery largely fails

Against The Virtual Brain's directed 76-region tract-tracing connectome, pyPTE
separates one-way edges from unconnected pairs at **AUC 0.62** — above chance,
but far from the 0.96–1.00 the same estimator reaches on sparse synthetic
networks. At 27% density nearly every region pair is joined by a short indirect
path, which is the previous caveat operating at whole-brain scale.

Scale bites twice. With `m` regions there are `m * (m - 1) / 2` channel pairs —
2,850 here — so the strictest FDR threshold falls below `2e-5`, while `N` surrogates cannot
produce a p-value below `1 / (N + 1)`. Edgewise surrogate testing at this scale
needs thousands of surrogates; `cluster_permutation` is the practical
alternative.

→ [`tvb_connectome.py`](https://github.com/patrk/pyPTE/blob/main/examples/tvb_connectome.py)

## Detection is model-dependent

There is no universal sample requirement. Phase oscillators resolve in a few
thousand samples; realistic Jansen-Rit columns needed tens of seconds, and were
strongly sensitive to the transmission delay. Effect size depends on the
dynamics, the SNR and the lag, which is why group-level testing is the robust
answer rather than a rule of thumb.

## Other things to know

- **Very strong coupling reduces detectability.** Oscillators synchronise and
  their phases become redundant, so dPTE peaks at intermediate coupling.
- **Common drivers** create apparent links: if `a` drives both `b` and `c`, then
  `b` and `c` will look connected.
- **Volume conduction** in sensor-space M/EEG produces zero-lag mixing that no
  phase-based measure can separate from genuine coupling. Prefer source space.
- **Cost is `O(m^2)`** in channels. A 100-channel montage over 8,000 samples
  takes about 0.8 s; 100 surrogates on it takes about 1.4 minutes.
