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

Filter to a band of interest **before** calling `PTE`. Phase is only meaningful
for a reasonably narrowband signal, and the measure says nothing about which
frequency an interaction lives at — that comes from how you filter.

## The two free choices, and why they are parameters

Steps 2 and 3 are not determined by the method, and both change the answer:

```python
PTE(signal, binning="scott", delay="zero-crossing")  # the defaults
```

**`binning`** accepts `"scott"` (default), `"auto"`, `"hillebrand"`, or an
integer bin count. Worth knowing: Hillebrand et al. 2016 specify
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
about 0 bits. Don't use that setting for dPTE, where it blurs phase and weakens
detection.

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
