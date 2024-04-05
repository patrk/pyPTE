from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Tuple

import numpy.typing as npt

from pyPTE.core import pyPTE


def _PTE_process(item: Tuple[Any, npt.ArrayLike]) ->(
        Dict[Any, Tuple[npt.NDArray, npt.NDArray]]):
    """
    Multi processing pool worker for PTE computation - wraps PTE method

    Parameters
    ----------
    item : tuple
        tuple: key, mne_raw object

    Returns
    -------
    result : dict
        dict: key, (dPTE, rawPTE) tuple

    """

    key, value = item
    print(key)
    (dPTE, raw_PTE) = pyPTE.PTE(value)
    result = {key: (dPTE, raw_PTE)}
    return result


def multi_process(measurements: Dict[Any, npt.ArrayLike]) -> (
        List[Dict[Any, Tuple[npt.NDArray, npt.NDArray]]]):
    """

    Parameters
    ----------
    measurements : dict
        dict : key, mne_raw object

    Returns
    -------
    results : dict
        dict: key, (dPTE, rawPTE) tuple

    """
    with Pool(processes=cpu_count()) as pool:
        result_dicts = pool.map(_PTE_process, list(measurements.items()))
    return result_dicts
