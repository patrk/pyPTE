# pyPTE examples

Each example builds a system whose directed connectivity is known in advance,
runs pyPTE on it, and **asserts** that the result matches the ground truth. They
are written to be run, not just read: if an assertion fails, the claim it makes
is no longer true.

```bash
uv sync --group examples
uv run python -m examples.two_node_coupling
```

Pass `--quick` to any of them for a faster, lower-resolution run. Figures are
written to `examples/figures/`.

## Suggested order

### 1. `two_node_coupling.py` — does it recover direction at all?

One oscillator drives another with a fixed lag. Sweeps coupling strength and
lag, confirming dPTE rises above 0.5 in the driving direction, falls below in
reverse, and sits at 0.5 when the two are independent.

The fourth panel is the counterweight and the most important part: two
**independent** oscillators differing only in signal-to-noise ratio produce a
dPTE around 0.73. The noisier channel is less predictable from its own past, so
the cleaner one appears to drive it. Since real M/EEG channels always differ in
SNR, raw dPTE values are not interpretable on their own.

### 2. `significance_testing.py` — what fixes that

Opens up `surrogate_test` rather than just using it. Three channels: A really
drives C, while B is independent but far noisier. In the raw matrix the artefact
A→B (0.719) scores *higher* than the real coupling A→C (0.664). Against their
own nulls they separate cleanly — A→B sits dead centre of its surrogate
distribution and is rejected at p = 0.74, A→C sits outside its own and survives
at p = 0.005.

Also draws the Benjamini-Hochberg cutoff, so the correction is visible rather
than implied.

Note a practical constraint it demonstrates: with `n` surrogates no p-value can
fall below `1 / (n + 1)`, so too few surrogates makes every pair uncallable
regardless of effect size.

### 3. `kuramoto_network.py` — recovering a known graph

Delayed Kuramoto oscillators in two topologies: a directed ring, and two
symmetrically-wired groups joined by one-way bridges. Scored with AUC (0.96 and
1.00), which needs no threshold, because the recovered matrix is dense while the
truth is sparse.

The two-group case makes a distinction that is easy to get wrong:
**dPTE measures direction, not the presence of coupling.** The strongly but
*symmetrically* coupled pairs inside each group sit at exactly 0.5. That is the
correct answer, not a miss — there is no net direction to find.

### 4. `neural_mass_network.py` — how much data, and what more of it buys

Jansen-Rit cortical columns producing a genuine alpha rhythm (verified at
~11 Hz with essentially all power in the 8–13 Hz band), coupled with a
transmission delay.

Recall reaches 1.0 by 60 s and saturates; precision then *falls*, from 0.50 to
0.36 by 200 s, as indirect paths become detectable. Those extra detections are
not errors — information genuinely flows that way — but they are not edges of
the network you were trying to recover. There is an optimum recording length for
network reconstruction.

### 5. `epoched_analysis.py` — the pipeline you would actually run

Two conditions, many short epochs, group statistics, both corrections. Shows
that 25 one-second epochs resolve coupling that a comparable stretch of
continuous recording leaves marginal: pooling over short epochs is the method,
not a compromise.

It also quantifies the central limitation. In a chain `a → b → c → d` where only
one-hop links exist:

| separation | mean dPTE | real edge? |
|---|---:|---|
| 1 hop | 0.582 | **yes** |
| 2 hops | 0.634 | no |
| 3 hops | **0.702** | no |

dPTE rises with path length, so ranking pairs by it returns close to the inverse
of the true network. pyPTE recovers the flow, not the wiring.

### 6. `tvb_connectome.py` — the same question on real anatomy

Everything above builds its own ground truth. This one uses The Virtual Brain's
bundled 76-region connectome, derived from tract-tracing rather than diffusion
imaging, so the directionality is genuine anatomy: 268 of its 1,560 edges exist
in one direction only, while tract lengths are correctly symmetric.

The result is largely negative, which is why it is worth running. pyPTE
separates one-way edges from unconnected pairs at **AUC 0.62**, against 0.96–1.00
for the sparse synthetic networks above. At 27% density nearly every region pair
is joined by a short indirect path — the same limitation as example 5, at whole-
brain scale.

It also runs into an arithmetic wall: 5,700 ordered pairs put the strictest FDR
threshold below `1e-5`, while `N` surrogates cannot yield a p-value below
`1 / (N + 1)`, so nothing survives edgewise testing at a practical surrogate
count. Use `cluster_permutation` at this scale.

Needs the optional TVB group, which is kept separate because `tvb-data` is a
~50 MB download:

```bash
uv sync --group examples --group tvb
uv run python -m examples.tvb_connectome
```

Note that TVB's `JansenRit` diverges to NaN on this connectome at default
coupling, so the example uses `Generic2dOscillator`.

## Shared code

- `models/kuramoto.py` — delayed Kuramoto oscillators with arbitrary directed
  coupling, plus topology builders: `ring_with_shortcut`, `two_groups`,
  `global_coupling` (mean-field) and `local_coupling` (nearest-neighbour). The
  symmetric variants are useful as negative controls, since a directional
  measure should correctly find nothing in them.
- `models/neural_mass_model.py` — coupled Jansen-Rit columns, integrated with
  stochastic Heun and a delay ring buffer.
- `utils/plotting.py` — shared figure style and a connectivity-matrix helper
  that puts row 0 at the top, since plotting defaults would flip a connectivity
  matrix top to bottom.

Both models use their own Euler-Maruyama/Heun integrators rather than an SDE
library. `sdeint` cannot integrate delay equations at all — its `f(y, t)` has no
access to history — and the delay is what makes coupling directional.

## Removed examples

The previous `kuramoto_global.py`, `kuramoto_local.py`, `mne_demo.py` and
`utils/stats.py` were deleted rather than repaired. None of them ran: they used
the pre-2024 axis convention, called `DataFrame.as_matrix()` (removed in pandas
1.0), or imported a module that has never been importable on Python 3.

Their ideas survive elsewhere. The global/local coupling distinction is now
`global_coupling` and `local_coupling` in `models/kuramoto.py`; the condition
contrast that `utils/stats.py` attempted is now `pyPTE.group_contrast`, without
the inverted Bonferroni correction and flipped significance mask it carried.
