from pyPTE.core.pyPTE import PTE, PTE_segments
from pyPTE.stats import (
    ClusterResult,
    GroupResult,
    SignificanceResult,
    benjamini_hochberg,
    cluster_permutation,
    group_contrast,
    group_test,
    surrogate_test,
    surrogates_needed,
)

__version__ = "1.14.1"

__all__ = [
    "PTE",
    "PTE_segments",
    "ClusterResult",
    "GroupResult",
    "SignificanceResult",
    "benjamini_hochberg",
    "cluster_permutation",
    "group_contrast",
    "group_test",
    "surrogate_test",
    "surrogates_needed",
    "__version__",
]
