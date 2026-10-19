# Running a real analysis

Reporting raw `dPTE` values is the single most common way to get a wrong answer
out of this library. Two independent signals that merely differ in
signal-to-noise ratio produce `dPTE` around `0.73` — indistinguishable from
genuine coupling. The value has to be tested against a null.

Which test depends on the shape of your data.

## One long continuous recording

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

Surrogates dominate the run time on large montages. `n_jobs=4` spreads them over
four processes (`-1` uses every CPU) without changing the result, since the
shifts are drawn from `seed` up front. In a script, keep the call under
`if __name__ == "__main__":`, which worker processes need on macOS and Windows.

## Many short epochs, trials or subjects — usually better

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

With MNE, `PTE_from_epochs` does the loop and estimates the delay and bins once
over all epochs, so every matrix is computed the same way:

```python
from pyPTE.adapters.mne_adapter import PTE_from_epochs

dPTE, raw_PTE = PTE_from_epochs(epochs, picks="eeg")  # (n_epochs, m, m) each
group_test(dPTE)
```

A single epoch may be far too short for a reliable estimate on its own; what
matters is that the estimate is unbiased, so noise averages out across
observations. Wilcoxon signed-rank is the default, matching the convention in
the M/EEG literature.

## Correcting for multiple comparisons

An `m`-channel recording produces `m * (m - 1)` ordered pairs — 9,900 for 100
channels, or 4,950 channel pairs, since dPTE in one direction fixes the other. Two corrections are provided, answering different questions:

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

## The recommended pipeline

```
band-pass filter  ->  epoch  ->  PTE per epoch  ->  group test
                                              ->  FDR or cluster correction
                                              ->  interpret only what survives
```
