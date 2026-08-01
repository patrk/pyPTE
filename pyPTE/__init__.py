from pyPTE.core.pyPTE import PTE
from pyPTE.stats import SignificanceResult, benjamini_hochberg, surrogate_test

__version__ = "1.1.0"

__all__ = [
    "PTE",
    "SignificanceResult",
    "benjamini_hochberg",
    "surrogate_test",
    "__version__",
]
