from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any

import numpy as np
import numpy.typing as npt

from pyPTE.core import pyPTE
from pyPTE.core.pyPTE import BinningRule, DelayRule


def _PTE_process(
    item: tuple[Any, npt.ArrayLike],
    binning: BinningRule = "scott",
    delay: DelayRule = "zero-crossing",
) -> dict[Any, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """
    Multi processing pool worker for PTE computation - wraps PTE method

    Parameters
    ----------
    item : tuple
        tuple: key, (n_channels, n_samples) array
    binning, delay
        passed to pyPTE.PTE

    Returns
    -------
    result : dict
        dict: key, (dPTE, rawPTE) tuple

    """

    key, value = item
    return {key: pyPTE.PTE(value, binning=binning, delay=delay)}


def multi_process(
    measurements: dict[Any, npt.ArrayLike],
    *,
    binning: BinningRule = "scott",
    delay: DelayRule = "zero-crossing",
    n_jobs: int | None = None,
) -> list[dict[Any, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]]:
    """
    Computes PTE for several measurements in parallel

    Parameters
    ----------
    measurements : dict
        dict : key, (n_channels, n_samples) array
    binning, delay
        passed to pyPTE.PTE for every measurement
    n_jobs : int | None
        worker processes; every CPU by default, 1 runs in this process

    Returns
    -------
    results : list of dict
        one dict per measurement: key, (dPTE, rawPTE) tuple

    """
    worker = partial(_PTE_process, binning=binning, delay=delay)
    items = list(measurements.items())
    if n_jobs == 1:
        return [worker(item) for item in items]
    with Pool(processes=n_jobs or cpu_count()) as pool:
        return pool.map(worker, items)
