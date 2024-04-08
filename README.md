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

## Features

- **Efficient Computation**: Utilizes optimized algorithms for fast computation of PTE values.
- **Flexible Input**: Accepts data in various formats, including raw time series and pre-processed frequency data.
- **Visualization Tools**: Integrated tools for visualizing the results of PTE analysis.
- **Extensive Documentation**: Comprehensive documentation available at [pypte.readthedocs.io](https://pypte.readthedocs.io).

## Installation and Setup

### Prerequisites:

**Essential Components:**
- Python (≥ 3.8)
- Git
- NumPy
- SciPy

**Additional Libraries (Optional):**
- mne-python
- pandas
- seaborn

**Environment Recommendations:**
To ensure a smooth installation and prevent potential module incompatibilities, I recommend using a virtual environment. Suitable options include:
- Conda
- pyenv

> Note: For those intending to use `mne-python`, an Anaconda3 setup is required.

## Deployment Steps

1. **Repository Cloning**
   Retrieve the `pyPTE` source code from the GitHub repository:
   ```bash
   git clone https://github.com/patrk/pyPTE.git
    ```

2. **Package Installation**
   Navigate to the pyPTE directory and initiate the installation:
   ```bash
   cd pyPTE
   python setup.py install
    ```

3. **Validation of Installation**
   Run the following commands to verify the installation of `pyPTE`:
   ```bash
    cd test
    py.test
    ```
   
## Usage
   Using `pyPTE` is straightforward. After importing the necessary functions, you can compute the PTE values for your data:
   ```python
   from pyPTE import compute_PTE 
   # Your data loading and preprocessing here
   PTE_values = compute_PTE(data)
   ```
    
## Contributing
Contributions to `pyPTE` are welcome! If you have suggestions, bug reports, or want to contribute code via Pull Requests

## License

`pyPTE` is released under the GPL-3.0 license. For more details, see the [LICENSE](https://github.com/patrk/pyPTE/blob/master/LICENSE) file.





