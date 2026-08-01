import pandas as pd

from pyPTE.core import pyPTE


def PTE_from_dataframe(data_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes dPTE and raw PTE matrices for the channels of a pandas.DataFrame

    Parameters
    ----------
    data_frame : pandas.DataFrame
        This object contains time-series data where pandas.DataFrame.index corresponds
        to the time samples and pandas.DataFrame.columns represents the
        individual channels

    Returns
    -------
    (dPTE_df, raw_PTE_df) : tuple of pandas.DataFrame objects
        The results from pyPTE.pyPTE.PTE are stored as pandas.DataFrames, indexed in
        both dimensions by pandas.DataFrame.columns of the input, so entry
        ``[i, j]`` describes information flow from channel ``i`` to channel ``j``

    """
    # the frame is time-major, while the core expects (n_channels, n_samples)
    time_series = data_frame.to_numpy().T
    dPTE, raw_PTE = pyPTE.PTE(time_series)

    channels = data_frame.columns
    dPTE_df = pd.DataFrame(dPTE, index=channels, columns=channels)
    raw_PTE_df = pd.DataFrame(raw_PTE, index=channels, columns=channels)
    return dPTE_df, raw_PTE_df
