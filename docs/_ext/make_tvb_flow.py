"""Regenerate docs/_data/tvb_flow.npz for the "Flow vs anatomy" tutorial.

The Virtual Brain is a heavy optional dependency and the simulation takes
minutes, so the docs build reads this cached result instead of running it.

    uv run --group tvb python docs/_ext/make_tvb_flow.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.tvb_connectome import flow, simulate, tracts  # noqa: E402

SECONDS = 120.0

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    brain, signal = simulate(SECONDS)
# band-passed and time-reversed, as the example runs it
dPTE = flow(signal)

out = ROOT / "docs" / "_data" / "tvb_flow.npz"
np.savez_compressed(
    out,
    # [i, j] is the tract from region i to j; TVB stores the transpose
    tracts=tracts(brain).astype(np.float32),
    dPTE=dPTE.astype(np.float32),
    labels=np.array([str(label) for label in brain.region_labels]),
    seconds=SECONDS,
)
print(f"wrote {out} from {signal.shape[1]} samples of {signal.shape[0]} regions")
