# What can fool it

Most of these numbers come from a script in `examples/` that asserts them.
The rest come from the simulations summarised in
[#109](https://github.com/patrk/pyPTE/issues/109), cited where they are used,
or from short simulations described next to them. "SNR" is the in-band
signal-to-noise ratio, as power, after the band-pass filter.

## Unequal signal-to-noise ratio fabricates direction

Two **completely independent** 10 Hz oscillators differing only in the
standard deviation of their white noise, 120 s at 250 Hz, mean of 10
recordings, **unfiltered** with Scott's bins and band-passed with
`binning="locking"`:

| noise A | noise B | dPTE[A→B], unfiltered | after 8–12 Hz |
|---:|---:|---:|---:|
| 0.2 | 0.2 | 0.50 | 0.49 |
| 0.2 | 0.4 | 0.69 | 0.48 |
| 0.2 | 0.8 | **0.80** | 0.54 |

The noisier channel is harder to predict from its own past, so the cleaner one
looks like a driver. Real recordings always differ in SNR between channels.

Band-passing shrinks it, and at these SNRs (420 against 26 in band at the
largest gap) almost removes it. A larger gap does not go away: in
`significance_testing.py`, channels at SNR 190 and 3 still give `0.72`
after the filter. A surrogate test usually rejects it,
because shifted surrogates keep each channel's noise and so carry the same
artefact: in 8 of 10 realisations, but it survived in the other two. Time
reversal rejected it in all ten, and confirmed the real coupling in the same
recordings in all ten. With `time_reversal=True` the differenced dPTE of the
independent pair stays at `0.49` at any noise ratio.

→ [`two_node_coupling.py`](https://github.com/patrk/pyPTE/blob/main/examples/two_node_coupling.py),
[`significance_testing.py`](https://github.com/patrk/pyPTE/blob/main/examples/significance_testing.py)

## Indirect paths outrank direct ones

In a chain `a → b → c → d` where only the one-hop links exist, 10 samples
(0.4 of a cycle) per hop, with equal noise on every channel (SNR about 70 at
`a`, falling to about 40 at `d`), band-passed, and time-reversal-differenced:

| separation | mean dPTE | real edge? |
|---|---:|---|
| 1 hop | 0.576 | **yes** |
| 2 hops | 0.625 | no |
| 3 hops | **0.659** | no |

`dPTE` rises monotonically with path length: a longer path is a longer lag,
and at this per-hop lag a longer lag carries more of the source's phase drift
to the far end. **Thresholding by dPTE returns close to the inverse of the
true network.** That ordering depends on the lag and the SNR: in the
simulations summarised in [#109](https://github.com/patrk/pyPTE/issues/109), with 16–25 samples per hop on
long recordings, the direct edges ranked first at noise 0.5 (SNR about 70),
though not at noise 1.0 (SNR about 17). If noise also builds
up along the chain, as it does when each node adds its own, the effect is
stronger still, but that part is the signal-to-noise artefact, and time
reversal removes it.

Indirect flow is the fundamental limit of any bivariate measure, and more data
does not get around it. In a ring of eight neural-mass columns, each of five
realisations went from finding nothing to finding every edge the right way
round somewhere between 15 and 60 s, and the two- and three-hop paths became
significant with the edges: precision was 0.5 at 15 s and a third from then
on. A significant dPTE identifies a directed path, not an edge; separating the
two needs a conditional measure
([#87](https://github.com/patrk/pyPTE/issues/87)).

→ [`epoched_analysis.py`](https://github.com/patrk/pyPTE/blob/main/examples/epoched_analysis.py),
[`neural_mass_network.py`](https://github.com/patrk/pyPTE/blob/main/examples/neural_mass_network.py)

## Symmetric coupling is not a no-direction control

Equal coupling weights both ways do not mean equal information flow. In the
Kuramoto example's two-group network, symmetrically coupled oscillators whose
natural frequencies differ (sd 0.5 Hz around 10 Hz) sit on average **0.11**
from 0.5, against 0.19 for the one-way bridges; with identical frequencies
the gap shrinks to 0.02. The sign follows neither the natural frequencies nor
which oscillator leads in phase.

Detuned symmetric pairs are significant against surrogates, too: 46 of 120
in the same network.

Near balance the direction is fragile: in the simulations summarised in
[#109](https://github.com/patrk/pyPTE/issues/109), a 5% change in one
direction's strength moved plain dPTE by about 0.1 either way, and a 4:3
bidirectional link read like a weak one-way link.

Averaging `dPTE` over both orientations of a pair cannot detect this:
`dPTE[i, j] + dPTE[j, i] = 1`, so that mean is 0.5 whatever the data. Judge a
pair by its distance from 0.5.

The same network shows the indirect-path limit above. A bridge's input
spreads through its tightly synchronised group, so every A → B pair carries
real flow: bridges outrank their own reverse with AUC 0.97 but outrank those
indirect pairs only 0.67 ± 0.18 of the time, over 20 realisations.

→ [`kuramoto_network.py`](https://github.com/patrk/pyPTE/blob/main/examples/kuramoto_network.py)

## A channel with mixed phase reads as a receiver

If one channel carries two independent rhythms in the band and the other
carries one, the mixed channel's phase is harder to predict from its own past,
and plain dPTE reads it as driven. With **no coupling at all**, one 10 Hz
rhythm against the sum of two independent ones, 8–12 Hz, `binning="locking"`:
plain dPTE was 0.74–0.75 towards the mixed channel in 20 recordings of 20, at
SNR 190 and at 17. Time reversal gave 0.50. In real data that is a channel
picking up two sources.

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
dynamics, the SNR and the lag, so test against a null on your own data rather
than trusting a rule of thumb; which test has the power for your data is
compared in [Running a real analysis](analysis.md).

## Other things to know

- **Very strong coupling reduces detectability.** Oscillators synchronise and
  their phases become redundant, so dPTE peaks at intermediate coupling.
- **Common drivers** create apparent links: if `a` drives both `b` and `c`, then
  `b` and `c` will look connected, the nearer one driving the farther. Time
  reversal does not help, since that order really is in the data. One rhythm
  reaching two channels 12 samples apart is exactly the coupled pair in
  [Running a real analysis](analysis.md), and at SNR 17 the spurious edge
  passed `PTE_segments` with time reversal and `group_test` in 23 of 40
  recordings and `reversal_confirmed` in 37.
- **Volume conduction** in sensor-space M/EEG produces zero-lag mixing that no
  phase-based measure can separate from genuine coupling. Prefer source space.
- **Cost is `O(m^2)`** in channels. A 100-channel montage over 8,000 samples
  takes about 0.8 s; 100 surrogates on it takes about 1.4 minutes.
