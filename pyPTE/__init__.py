from pyPTE.core.pyPTE import PTE
from pyPTE.stats import (
    GroupResult,
    SignificanceResult,
    benjamini_hochberg,
    group_contrast,
    group_test,
    surrogate_test,
)

__version__ = "1.1.0"

__all__ = [
    "PTE",
    "GroupResult",
    "SignificanceResult",
    "benjamini_hochberg",
    "group_contrast",
    "group_test",
    "surrogate_test",
    "__version__",
]
