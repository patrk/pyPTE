# pyPTE: Phase Transfer Entropy in Python

`pyPTE` is an open-source Python implementation of the Phase Transfer Entropy method, designed to analyze directed connectivity in networks influenced by oscillatory interactions. This tool is inspired by the following foundational works:

- Lobier et al., 2014: [Phase transfer entropy: A novel phase-based measure for directed connectivity in networks coupled by oscillatory interactions](http://dx.doi.org/10.1016/j.neuroimage.2013.08.056).
- Hillebrand et al., 2016: [Direction of information flow in large-scale resting-state networks is frequency-dependent](http://dx.doi.org/10.1073/pnas.1515657113).

## Introduction

Phase Transfer Entropy (PTE) is a measure for directed connectivity in networks coupled by oscillatory interactions. The `pyPTE` library provides a Python implementation of this method, allowing researchers and developers to apply PTE analysis to their data.

### Mathematical Background

The mathematical formulation of PTE can be described as follows:

Given two time series $`X`$ and $`Y`$, the PTE is defined as:

$$PTE_{X \to Y} = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)$$

The PTE value quantifies the amount of uncertainty reduced in predicting the future phase of $`Y`$ when considering the current phase of $`X`$.

## Contributing
Contributions to `pyPTE` are welcome! If you have suggestions, bug reports, or want to contribute code via Pull Requests

## License

`pyPTE` is released under the GPL-3.0 license. For more details, see the [LICENSE](https://github.com/patrk/pyPTE/blob/master/LICENSE) file.





