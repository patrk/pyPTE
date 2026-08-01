from pyPTE.core.pyPTE import PTE
from pyPTE.stats import (
    ClusterResult,
    GroupResult,
    SignificanceResult,
    benjamini_hochberg,
    cluster_permutation,
    group_contrast,
    group_test,
    surrogate_test,
)

__version__ = "1.2.0"

__all__ = [
    "PTE",
    "ClusterResult",
    "GroupResult",
    "SignificanceResult",
    "benjamini_hochberg",
    "cluster_permutation",
    "group_contrast",
    "group_test",
    "surrogate_test",
    "__version__",
]
