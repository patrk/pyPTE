"""
Flow vs anatomy at brain scale
==============================

Whole-brain activity simulated with The Virtual Brain on its 76-region
connectome, which is directed: it comes from tract tracing, so many
connections exist in one direction only. If dPTE recovered anatomy, its
strongest flows would run along those one-way tracts.

The simulation needs the optional ``tvb`` dependencies and is too heavy to
run on every documentation build, so this page reads a cached result made by
``docs/_ext/make_tvb_flow.py``.
"""

# %%
# How often does a strong flow sit on a real tract?
# -------------------------------------------------
# Rank every ordered pair by dPTE and ask how well that ranking separates
# one-way tracts from unconnected pairs. An AUC of 0.5 is chance; the same
# estimator reaches 0.96 to 1.00 on the sparse synthetic networks of the
# tutorial.

from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from examples.utils import style, teaching

style.use()
cache = np.load(Path("..") / "_data" / "tvb_flow.npz")
weights, dPTE, labels = cache["weights"], cache["dPTE"], list(cache["labels"])

off_diagonal = ~np.eye(weights.shape[0], dtype=bool)
one_way = (weights > 0) & (weights.T == 0)
unconnected = off_diagonal & (weights == 0) & (weights.T == 0)
scores = np.concatenate([dPTE[one_way], dPTE[unconnected]])
ranks = rankdata(scores)
n_pos, n_neg = int(one_way.sum()), int(unconnected.sum())
auc = (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
density = (weights > 0).sum() / off_diagonal.sum()
print(
    f"{weights.shape[0]} regions, {density:.0%} of pairs connected, "
    f"{n_pos} one-way tracts, AUC {auc:.2f}"
)

# %%
# The strongest flows, drawn
# --------------------------
# Regions sit around a circle, each hemisphere on its own side. **a** shows
# the 60 strongest one-way tracts, mostly within a hemisphere. **b** shows the
# 60 strongest dPTE flows, in red where a one-way tract runs the same way.
#
# With most region pairs connected, nearly every pair is also joined by a
# short indirect route, and dPTE ranks those as highly as direct ones. Use it
# to ask which way activity flows, not to infer the wiring.

teaching.chord_pair(weights, dPTE, labels)
