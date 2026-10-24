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
# The activity was band-passed to its dominant rhythm and analysed with time
# reversal. Rank every ordered pair by dPTE and ask how well that ranking
# separates one-way tracts from unconnected pairs; an AUC of 0.5 is chance.
# Then rank by how far dPTE sits from 0.5, on either side, which asks only
# whether two regions are connected.

from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from examples.utils import style, teaching

style.use()
cache = np.load(Path("..") / "_data" / "tvb_flow.npz")
# [i, j] is the tract from region i to region j
tracts, dPTE, labels = cache["tracts"], cache["dPTE"], list(cache["labels"])

off_diagonal = ~np.eye(tracts.shape[0], dtype=bool)
one_way = (tracts > 0) & (tracts.T == 0)
unconnected = off_diagonal & (tracts == 0) & (tracts.T == 0)
n_pos, n_neg = int(one_way.sum()), int(unconnected.sum())


def auc(score):
    ranks = rankdata(np.concatenate([score[one_way], score[unconnected]]))
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


density = (tracts > 0).sum() / off_diagonal.sum()
print(
    f"{tracts.shape[0]} regions, {density:.0%} of pairs connected, "
    f"{n_pos} one-way tracts\n"
    f"AUC by direction {auc(dPTE):.2f}, by distance from 0.5 {auc(abs(dPTE - 0.5)):.2f}"
)

# %%
# The strongest flows, drawn
# --------------------------
# Regions sit around a circle, each hemisphere on its own side. **a** shows
# the 60 strongest one-way tracts, mostly within a hemisphere. **b** shows the
# 60 strongest dPTE flows, in red where a one-way tract runs the same way.
#
# dPTE sees that two regions are connected far better than which way the
# tract runs. With most region pairs connected, nearly every pair is also
# joined by short indirect routes, which can run either way. Use it to ask
# which way activity flows, not to infer the wiring.

teaching.chord_pair(tracts, dPTE, labels)
