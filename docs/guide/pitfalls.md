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

Band-passing shrinks it but does not remove it: after an 8–12 Hz filter, an
eight-fold noise gap still gives `0.72`. A surrogate test usually rejects it,
because shifted surrogates keep each channel's noise and so carry the same
artefact: in 8 of 10 realisations, but it survived in the other two. Time
reversal rejected it in all ten, and confirmed the real coupling in the same
recordings in all ten. With `time_reversal=True` the differenced dPTE of the
independent pair stays at `0.49` at any noise ratio.

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
does not get around it. In a ring of eight neural-mass columns, each of five
realisations went from finding nothing to finding every edge the right way
round somewhere between 15 and 60 s, and the two- and three-hop paths became
significant with the edges: precision was 0.5 at 15 s and a third from then
on. A significant dPTE identifies a directed path, not an edge; separating the
two needs a conditional measure
([#87](https://github.com/patrk/pyPTE/issues/87)).

→ [`epoched_analysis.py`](https://github.com/patrk/pyPTE/blob/main/examples/epoched_analysis.py),
[`neural_mass_network.py`](https://github.com/patrk/pyPTE/blob/main/examples/neural_mass_network.py)

## Symmetric coupling is invisible — by design

Strongly but *symmetrically* coupled channels sit at exactly `0.500`. That is
the correct answer, not a miss: `dPTE` measures **net** direction, and a
balanced bidirectional link has none. Score it against raw connectivity and you
will penalise it for being right.

→ [`kuramoto_network.py`](https://github.com/patrk/pyPTE/blob/main/examples/kuramoto_network.py)

## On a real connectome, structural recovery largely fails

Against The Virtual Brain's directed 76-region tract-tracing connectome,
band-passed and time-reversed dPTE sees *that* two regions are connected but
not which way. How far it sits from 0.5, on either side, separates one-way
tracts from unconnected pairs at **AUC 0.78**; its sign does so at **0.53**,
barely above chance. At 27% density nearly every region pair is also joined by
short indirect paths, which is the previous caveat operating at whole-brain
scale.

Plain dPTE scores **0.39**, below chance. Regions that receive more input
oscillate more strongly (rank correlation 1.0), so their rhythm is cleaner,
and dPTE reads the cleaner channel as the driver: the signal-to-noise artefact
again, here pointing against the anatomy. An earlier version of this page
reported AUC 0.62; that came from the same artefact, scored against the
connectome transposed. TVB stores its weights as `[to, from]`
([#106](https://github.com/patrk/pyPTE/issues/106)).

Scale bites twice. With `m` regions there are `m * (m - 1) / 2` channel pairs —
2,850 here — so the strictest FDR threshold falls below `2e-5`, while `N` surrogates cannot
produce a p-value below `1 / (N + 1)`. Edgewise surrogate testing at this scale
needs thousands of surrogates; `cluster_permutation` is the practical
alternative.

→ [`tvb_connectome.py`](https://github.com/patrk/pyPTE/blob/main/examples/tvb_connectome.py)

## Detection is model-dependent

There is no universal sample requirement. Phase oscillators resolve in a few
thousand samples; realistic Jansen-Rit columns needed 15 to 60 s, and were
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
