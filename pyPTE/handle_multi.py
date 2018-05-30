from multiprocessing import Pool, cpu_count
from pyPTE import pyPTE

def _PTE_process(item):
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
    # dPTE, rawPTE = phase_transfer_entropy(value, method='myPTE')
    raw_PTE = pyPTE.PTE(value, method='myPTE')
    result = dict()
    # result[key] = dPTE, rawPTE
    result[key] = raw_PTE
    return result


def multi_process(measurements):
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
    results = []
    print(cpu_count())
    pool = Pool(processes=cpu_count())
    r = pool.map_async(_PTE_process, list(measurements.items()), callback=results.append)
    r.wait()
    pool.close()
    return results