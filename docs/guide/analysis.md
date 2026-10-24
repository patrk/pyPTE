# Running a real analysis

Reporting raw `dPTE` values is the single most common way to get a wrong answer
out of this library. Two independent signals that merely differ in
signal-to-noise ratio produce `dPTE` around `0.73` — indistinguishable from
genuine coupling. The value has to be tested against a null.

```{warning}
**Band-pass to the rhythm of interest first, and narrowly.** On broadband
data dPTE points from the cleaner channel to the noisier one, whatever the
coupling ([#97](https://github.com/patrk/pyPTE/issues/97)). In simulation, a
noisy 10 Hz driver coupled to a clean target, 12 samples later at 250 Hz,
comes out the right way round this often (12 recordings of 32 s each):

| preprocessing | right way round |
|---|---|
| none, or 1–45 Hz | 0% |
| 4–16 Hz or 6–14 Hz | 0% |
| 8–12 Hz | 92% |
| 8–12 Hz, driver twice as noisy | 8% |
| 8–12 Hz, `binning="hillebrand"` | 8% |

Filtering helps only while the in-band signal-to-noise gap is modest, and the
`hillebrand` bin rule gets the sign wrong even then. For a direction you can
rely on, use the time-reversal check below.
```

```{warning}
**Use enough bins.** With too few phase bins, the direction dPTE reports for
a real coupling depends on how many bins the phase advances over the delay,
and can come out significantly backwards
([#99](https://github.com/patrk/pyPTE/issues/99),
[#110](https://github.com/patrk/pyPTE/issues/110)). How many is enough
depends on how tightly the two channels' phases lock: π / σ, where σ is the
spread of their phase difference. Loosely locked pairs need about 16; very
clean, tightly locked ones up to 64, and no fixed count is safe for both.
`binning="locking"` chooses that count from the data, and every rule warns
when it gives fewer. `PTE_segments` and `PTE_from_epochs` use it by default;
`PTE` keeps Scott's rule.
```

Which test depends on the shape of your data.

## One long continuous recording

Compare the observed matrix against **time-shifted surrogates**: each channel is
circularly shifted by its own random offset, at least 5% of the recording away
from every other channel's, which preserves every per-channel property exactly
while destroying cross-channel timing. Any apparent
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

A significant pair can still point the wrong way. A difference in
signal-to-noise ratio between two coupled channels, or one source mixed
instantaneously into two channels with different noise, gives a direction the
surrogate test calls significant. The surrogates rule out independence, not a
wrong direction.

The check borrowed from Granger causality is to reverse time
([Haufe et al. 2012](https://doi.org/10.1007/978-3-642-28551-6_4);
[Winkler et al. 2016](https://doi.org/10.1109/tsp.2016.2531628)). A real
direction reverses with the data; one made by noise or mixing stays where it
was. pyPTE offers it two ways:

```python
# one continuous recording, conservative
result = surrogate_test(signal, n_surrogates=200, seed=0, time_reversal=True)
result.reversal_confirmed  # significant both ways round, with opposite signs

# more power: segments or epochs, differenced, tested for consistency
from pyPTE import PTE_segments

dPTE, _ = PTE_segments(signal, 10, time_reversal=True)
group_test(dPTE).significant
```

With `time_reversal=True`, `PTE`, `PTE_segments` and `PTE_from_epochs` return
the *differenced* dPTE, `0.5 + (dPTE - dPTE_reversed) / 2`. Directions from
noise or mixing cancel in it and real ones remain. `group_test` then asks
whether that is consistent across segments, which is the right null for it.
The surrogate test is not: its null assumes independent channels, so under
mixing it calls random differenced values significant. `surrogate_test`
reports `differenced_dPTE` only as an effect size.

How the three compare on simulated 10 Hz recordings of 120 s each, band-passed
to 8–12 Hz, with the coupling 12 samples long, close to the estimated delay
(20 recordings per cell; the segment test's false positives over 60):

| | false direction, mixing | noisy driver found | twice as noisy found | equal noise found |
|---|---|---|---|---|
| `surrogate_test`, plain `significant` | 19/20 | 15/20 | 0/20 (13/20 backwards) | 20/20 |
| `reversal_confirmed` | 0/20 | 15/20 | 0/20 | 20/20 |
| `PTE_segments` + `group_test`, 10 × 12 s | 1/60 | 18/20 | 6/20 | 20/20 |

The noise levels are broadband standard deviations of 1.0 for the noisy
driver, 2.0 for twice as noisy, 0.1 for the clean target and 0.3 for equal
noise; after the 8–12 Hz filter, 0.3 leaves an in-band signal-to-noise ratio
of about 190, far cleaner than real recordings. At equal noise of 1.0, an
in-band ratio of about 17, the segment test finds the coupling in 6 of 20.
With Scott's rule the segments get 15 bins, and the segment test then called
the equal-noise coupling significant *backwards* in 8 recordings out of 20.
Where PTE's direction is reliable is mapped in
[#85](https://github.com/patrk/pyPTE/issues/85).

- **Enough segments.** Wilcoxon over n segments cannot give a p-value below
  2 / 2ⁿ, and Benjamini-Hochberg over P pairs needs α / P, so n must be at
  least log₂(2P / α): 6 for two channels, 11 for ten, 17 for 64. `group_test`
  warns when it cannot reach significance.
- **Enough bins per segment.** Keep the default `binning="locking"`. Scott's
  rule sizes bins for one segment and gives too few on short ones.
- **Independent segments.** Segments cut from one recording share its slow
  drift, so they are not fully independent and the test is somewhat
  optimistic. Separate trials or recordings do not have that problem.
- **Band-pass anyway.** On broadband data the differenced value is close to
  0.5 for everything: it no longer points the wrong way, but it finds nothing.
- **Not proven, shown.** The cancellation is proven for linear models
  (Winkler et al.); nonlinear coupling can survive time reversal
  ([Kořenek & Hlinka 2021](https://doi.org/10.3390/e23081067)). The table is
  the evidence for PTE.
- **Common drivers.** Neither check catches a common driver that reaches the
  two channels at different lags: that order really is in the data.

## Epochs, trials or subjects

When the data come as repeated observations, `group_test` asks whether a
direction is consistent across them, and `group_contrast` whether it differs
between conditions. Three settings decide whether the answer can be trusted.

```python
from pyPTE import group_test
from pyPTE.adapters.mne_adapter import PTE_from_epochs

dPTE, _ = PTE_from_epochs(epochs, picks="eeg", time_reversal=True)
group_test(dPTE)  # (n_epochs, m, m) in, one test per pair
```

- **`time_reversal=True`, always.** A difference in signal-to-noise ratio or
  instantaneous mixing pushes dPTE the same way in every epoch, which is
  exactly the consistency `group_test` looks for. In simulation, plain dPTE
  over 1 s epochs reported mixing as a significant direction in 8 to 10 runs
  out of 10, and paired a white-noise channel with rhythmic ones in 8 of 8.
  Time reversal cancels both.
- **The default bins, not Scott's rule.** Scott's rule sizes bins for one
  epoch and gives about 7 on a 1 s epoch, where the direction follows the
  delay rather than the data: 240 one-second epochs reported the coupling
  below as significantly *backwards* in 20 runs out of 20. The default,
  `binning="locking"`, pools each pair's phase locking over the epochs.
- **Long epochs, or many.** Short epochs carry little directional information.
  Simulated coupling of 12 samples at 250 Hz with equal noise of 1.0 (in-band
  signal-to-noise ratio about 17), 8–12 Hz, `time_reversal=True`, 20 runs
  each:

  | epochs | found the right way | wrong way |
  |---|---|---|
  | 60 × 1 s | 1 | 0 |
  | 240 × 1 s | 6 | 0 |
  | 60 × 4 s | 18 | 0 |
  | 30 × 8 s | 18 | 0 |

`PTE_from_epochs` estimates the delay and bins once over all epochs, so every
matrix is computed the same way. Band-pass the continuous recording before
epoching rather than each epoch, so filter edges stay out of the epochs.

**Evoked responses** need care. A response phase-locked to the stimulus gives
two channels the same phase offset in every trial, which reads as a consistent
direction; if the latencies differ only because of a common input, it is not
an interaction. Subtract the evoked response first to analyse induced activity.

Earlier versions of this guide called short epochs "far more sensitive" than a
continuous recording. That came from simulations in which the driver was also
the cleaner channel and every rhythm started at the same phase, and it does
not hold ([#99](https://github.com/patrk/pyPTE/issues/99)).

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
