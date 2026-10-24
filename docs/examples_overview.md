# Example index and related work

The examples run as part of this site, see [the gallery](examples/index).

Every example builds a system with known ground truth, runs `pyPTE`, and
**asserts** the result. If an assertion fails, the claim it makes is no longer
true.

```bash
uv sync --group examples
uv run python -m examples.two_node_coupling      # add --quick for a faster run
```

| example | what it establishes |
|---|---|
| [`two_node_coupling`](https://github.com/patrk/pyPTE/blob/main/examples/two_node_coupling.py) | direction is recovered; SNR asymmetry fakes it |
| [`kuramoto_network`](https://github.com/patrk/pyPTE/blob/main/examples/kuramoto_network.py) | known directed graphs recovered, AUC 0.96–1.00 |
| [`neural_mass_network`](https://github.com/patrk/pyPTE/blob/main/examples/neural_mass_network.py) | recording length, and why indirect paths come with the edges, on Jansen-Rit columns |
| [`significance_testing`](https://github.com/patrk/pyPTE/blob/main/examples/significance_testing.py) | what a surrogate test does, drawn out |
| [`epoched_analysis`](https://github.com/patrk/pyPTE/blob/main/examples/epoched_analysis.py) | full epoched two-condition pipeline |
| [`tvb_connectome`](https://github.com/patrk/pyPTE/blob/main/examples/tvb_connectome.py) | a real directed connectome, where recovery mostly fails |

See [`examples/README.md`](https://github.com/patrk/pyPTE/blob/main/examples/README.md) for details.

## Related work

The example models here are self-contained on purpose, so they run in CI without
extra dependencies. If you want richer or more citable ground-truth signals,
these are the simulators worth knowing about:

- **[The Virtual Brain](https://github.com/the-virtual-brain/tvb-root)**
  (`tvb-library`, GPL-3.0-or-later) — the reference whole-brain platform, and
  the one worth reaching for first. It ships a genuinely **directed** 76-region
  connectome from tract-tracing rather than a symmetric DTI matrix, which is
  what a directional measure needs as ground truth, plus ~30 population models
  and EEG/MEG forward monitors. Sanz Leon et al., *Front. Neuroinform.* 7:10
  (2013).
  → **[`examples/tvb_connectome.py`](https://github.com/patrk/pyPTE/blob/main/examples/tvb_connectome.py)** runs pyPTE
  against that connectome, and is the most informative example here because the
  result is largely negative.
- **[PyRates](https://github.com/pyrates-neuroscience/PyRates)** (GPL-3.0) — a
  much smaller dependency footprint, with explicit per-edge directed coupling
  and delays. No bundled connectome, so you supply the topology, which is what
  the self-contained models here already do; the gain over them is citability
  rather than capability. Gast et al., *PLOS ONE* 14(12):e0225900 (2019).
- [neurolib](https://github.com/neurolib-dev/neurolib) (MIT) — whole-brain
  modelling with several neural mass models. **No example here**, deliberately:
  its bundled HCP connectome is symmetric, so it carries no net direction for a
  directional measure to recover, and the project has had little activity since
  December 2024.
- [brainmass](https://github.com/chaobrain/brainmass) (Apache-2.0), part of the
  [BrainX](https://brainx.chaobrain.com/) ecosystem — differentiable neural mass
  models on JAX. **No example here**, deliberately: at v0.1.1 the API is still
  moving, and it pulls in the whole JAX stack for a demonstration the models
  above already provide.

`pyPTE` depends on none of them.
