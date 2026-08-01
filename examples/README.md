# pyPTE examples

Each example builds a system whose directed connectivity is known in advance,
runs pyPTE on it, and **asserts** that the recovered result matches the ground
truth. They are written to be run, not just read: if an assertion fails, the
claim it makes is no longer true.

```bash
uv sync --group examples
uv run python -m examples.two_node_coupling
uv run python -m examples.kuramoto_network
```

Pass `--quick` to any of them for a faster, lower-resolution run. Figures are
written to `examples/figures/`.

## The examples

### `two_node_coupling.py`

One oscillator drives another with a fixed lag. Sweeps coupling strength and
lag, confirming dPTE rises above 0.5 in the driving direction, falls below it in
reverse, and sits at 0.5 when the two are independent.

The fourth panel is the counterweight, and the most important part: two
**independent** oscillators that differ only in signal-to-noise ratio produce a
dPTE around 0.73, comparable to genuine coupling. The noisier channel is less
predictable from its own past, so the cleaner one appears to drive it. Since
real M/EEG channels always differ in SNR, raw dPTE values are not interpretable
on their own.

### `kuramoto_network.py`

Delayed Kuramoto oscillators wired into two known topologies: a directed ring,
and two symmetrically-wired groups joined by one-way bridges. Scored with AUC,
which needs no threshold, because the recovered matrix is dense while the truth
is sparse.

The two-group case makes a distinction that is easy to get wrong:
**dPTE measures direction, not the presence of coupling.** The strongly but
*symmetrically* coupled pairs inside each group sit at exactly 0.5. That is the
correct answer, not a miss — there is no net direction to find. Scoring dPTE
against raw connectivity rather than net flow would penalise it for being right.

## What these examples do not yet do

Turning a dPTE matrix into a graph requires a significance test against
surrogate data plus correction for multiple comparisons, since every one of the
`m * m` pairs receives a score. That, along with the indirect-connection
problem — a bivariate measure reports `a -> c` when the truth is `a -> b -> c` —
is the subject of the proof-of-concept example.

## Removed examples

The previous `kuramoto_global.py`, `kuramoto_local.py`, `mne_demo.py`,
`utils/stats.py` and `models/neural_mass_model.py` were deleted rather than
repaired. None of them ran: they used the pre-2024 axis convention, called
`DataFrame.as_matrix()` (removed in pandas 1.0), or imported a module that has
never been importable on Python 3.

The Jansen-Rit neural mass model is worth restoring, but needs a real rewrite
rather than a patch — its coupling term read `C[i, :] * y[i, 6]`, which is node
`i`'s own output, so the masses never actually coupled; `m` was undefined
whenever `C` was `None`; and its two-dimensional state cannot be integrated by
`sdeint` at all. Bringing it back is tracked separately so it can be validated
against published alpha-band behaviour instead of merely made to run.
