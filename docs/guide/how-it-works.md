# How it works

For an illustrated walk through the same steps, see the
[tutorial](../tutorial/how_pte_works).

For each ordered channel pair, `PTE` does five things:

1. **Extract phase.** A Hilbert transform turns each channel into an analytic
   signal; the angle between its real and imaginary parts is the instantaneous
   phase, on `[-pi, pi)`.
2. **Estimate a delay.** The analysis lag is derived from the mean number of
   samples between zero crossings, so it adapts to the dominant rhythm.
3. **Bin the phase.** Scott's rule sets the bin width from the data by default;
   see below for what that choice costs and the alternatives.
4. **Count joint states.** Occurrences of `(Y_future, Y_now, X_now)` are counted
   and turned into entropies.
5. **Combine.** `PTE = H(Ypr,Y) + H(Y,X) - H(Y) - H(Ypr,Y,X)`, which is the
   conditional mutual information `I(Y_future ; X | Y)` — how much better you
   can predict `Y`'s next phase knowing `X`, beyond knowing `Y`'s own past.

Equivalently, in the usual notation:

$$PTE_{X \to Y} = H(Y_{t+\delta} \mid Y_t) - H(Y_{t+\delta} \mid Y_t, X_t)$$

Because it is a conditional mutual information, raw PTE is non-negative and
bounded above by `log2(n_bins)`. Both are asserted in the test suite.

## Where the direction comes from

A real rhythm's phase drifts: its frequency wanders a little from cycle to
cycle. When `X` drives `Y` with a lag, `Y` receives each of `X`'s wanderings
that much later, so `X`'s phase now says something about where `Y`'s phase
will be, and not the other way round. That asymmetry is what PTE measures.

The phase offset between the channels is not the signal. A perfectly periodic
rhythm has no drift, and two channels carrying it hold the same offset
forever, which predicts nothing in either direction. Nor is there an
ambiguity at a lag of half a cycle or a full cycle: in the simulations
summarised in [#109](https://github.com/patrk/pyPTE/issues/109) dPTE kept its sign
through both and peaked at a lag of one to two cycles, depending on how fast
the phase drifts and on the noise. How large the effect is for a given lag
depends on the same two things, so a statement like "a lag of 5 samples is
hard to detect" is about the drift and the SNR, not the lag alone.

Filter to a band of interest **before** calling `PTE`. Phase is only meaningful
for a reasonably narrowband signal, and the measure says nothing about which
frequency an interaction lives at — that comes from how you filter.

## The two free choices, and why they are parameters

Steps 2 and 3 are not determined by the method, and both change the answer:

```python
PTE(signal, binning="scott", delay="zero-crossing")  # the defaults
```

**`binning`** accepts `"scott"` (the default for `PTE`), `"locking"` (the
default for `PTE_segments` and `PTE_from_epochs`), `"auto"`, `"hillebrand"`,
or an integer bin count. `"locking"` sizes the bins by how tightly the
channels' phases lock, which is what keeps the direction of a real coupling
from aliasing; see [Running a real analysis](analysis.md).

Worth knowing: Hillebrand et al. 2016 specify
`exp(0.626 + 0.4·ln(N_s − δ − 1))`, which is **3–4× more bins than Scott's rule**
— 68 against 20 at 8,000 samples. At 68 bins the three-way histogram has
314,000 cells for 8,000 samples and is hopelessly sparse.

Scott's rule stays the default because dPTE depends on fine phase resolution,
but it leaves about one sample per cell of the three-way histogram. That biases
**raw PTE** upwards: two unrelated signals report around 0.75 bits. The bias
largely cancels in the dPTE ratio, so read dPTE as usual and never read raw PTE
on its own. When the size of raw PTE in bits matters, use
`PTE(signal, binning="auto", bias_correction=True)`: bins sized for the
three-way histogram plus the Miller–Madow correction bring unrelated signals to
about 0 bits. That gives few bins, 3 at 1,000 samples, 7 at 8,000 and 17 at
100,000, so use `"auto"` only for raw PTE, never for dPTE, where so few bins
blur the phase.

It is not a cosmetic difference: the effect size depends on the rule, and
which rule reports more changes with the recording length and the
signal-to-noise ratio. Mean plain dPTE for a coupled pair with equal noise,
band-passed to 8–12 Hz, 20 recordings each:

| samples | in-band SNR | Scott | Hillebrand | locking |
|---:|---:|---:|---:|---:|
| 8,000 | 190 | 0.65 | 0.71 | 0.82 |
| 16,000 | 190 | 0.83 | 0.69 | 0.86 |
| 8,000 | 17 | 0.58 | 0.51 | 0.57 |
| 30,000 | 17 | 0.65 | 0.51 | 0.58 |

Effect sizes from different rules are not comparable, and at a realistic SNR
Hillebrand's rule loses the coupling almost entirely; it can also point the
wrong way (see the band-pass warning in
[Running a real analysis](analysis.md)). Pass `binning="hillebrand"` only to
match the paper.

**`delay`** accepts `"zero-crossing"` (default, and what Hillebrand specifies),
`"phase-increment"`, or an integer. The two rules estimate the same quantity and
agree within a sample on narrowband signals; they diverge on broadband or
DC-offset signals, where neither is trustworthy. A signal containing several
rhythms gives a delay tracking the fastest one, which is another reason to
filter first.
