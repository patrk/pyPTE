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

### How much transfer, beyond chance

Raw PTE is a histogram estimate, and it is biased upwards: two independent
oscillators can score a quarter of a bit, and the floor moves with the bin
count, the recording length and each channel's spectrum. Comparing raw PTE
across recordings or settings partly compares that bias.

`result.effective_PTE` subtracts it. It is raw PTE minus the mean raw PTE of the
surrogates (`result.raw_null_mean`): the *effective transfer entropy* of
[Marschinski & Kantz (2002)](https://doi.org/10.1140/epjb/e2002-00379-2) with
phase as the variable, as in
[Yang et al. (2017)](https://doi.org/10.1016/j.physa.2016.10.085) and
[Kudo et al. (2022)](https://doi.org/10.1089/brain.2020.0989). The surrogates
come for free, since the test builds them anyway. Unlike those papers, which
shuffle the source, pyPTE shifts it in time, which keeps each channel's
narrow-band autocorrelation, the thing the bias depends on; shuffled phase is
white noise.

```python
result.effective_PTE  # bits beyond chance, near 0 without coupling
```

Negative values are kept rather than clipped: a pair with no coupling lands
either side of zero, and clipping would push the average over subjects above
it. Significance still comes from the dPTE test; effective PTE says how large an
effect is, not whether it is real.

### Does the direction survive time reversal?

A significant pair can still point the wrong way. On broadband data, dPTE
points from the cleaner channel to the noisier one: a noisy driver coupled to a
clean target comes out as the target driving the driver, and with enough data
the surrogate test calls that significant
([#97](https://github.com/patrk/pyPTE/issues/97)). Instantaneous mixing of one
source into two channels with different noise does the same.

The check borrowed from Granger causality is to reverse time
([Haufe et al. 2012](https://doi.org/10.1007/978-3-642-28551-6_4);
[Winkler et al. 2016](https://doi.org/10.1109/tsp.2016.2531628)). A real
direction reverses with the data; one made by noise or mixing stays where it
was.

```python
result = surrogate_test(signal, n_surrogates=200, seed=0, time_reversal=True)
result.reversal_confirmed  # significant, and significantly flipped when reversed
result.reversed_dPTE
```

A pair is confirmed only when it is significant in both directions of time
with opposite signs, which is the conservative rule. The reversed recording is
tested against the same shifts, with the forward delay and bins, so the run
takes twice as long. It cannot catch a common driver that reaches the two
channels at different lags: that order really is in the data, so it reverses
like a genuine interaction.

Band-pass filtering to the oscillation of interest first is what makes the
direction come out right in the first place. Time reversal tells you when it
has not.

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
