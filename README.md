# pyPTE: Phase Transfer Entropy in Python

`pyPTE` is an open-source Python implementation of the Phase Transfer Entropy method, designed to analyze directed connectivity in networks influenced by oscillatory interactions. This tool is inspired by the following foundational works:

- Lobier et al., 2014: [Phase transfer entropy: A novel phase-based measure for directed connectivity in networks coupled by oscillatory interactions](http://dx.doi.org/10.1016/j.neuroimage.2013.08.056).
- Hillebrand et al., 2016: [Direction of information flow in large-scale resting-state networks is frequency-dependent](http://dx.doi.org/10.1073/pnas.1515657113).

## Installation and Setup

### Prerequisites:

**Essential Components:**
- Python (≥ 3.6)
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
4. **Usage Examples**
    tbd



