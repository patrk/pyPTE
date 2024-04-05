import pandas as pd

from pyPTE.core import pyPTE


def PTE_from_dataframe(data_frame):
    """
    This is a wrapper which allows calculating dPTE,PTE matrices by passing an pandas.DataFrame

    Parameters
    ----------
    data_frame : pandas.DataFrame
        This object contains time-series data where pandas.DataFrame.index corresponds to the time samples and
        pandas.DataFrame.columns represents the individual channels

    Returns
    -------
    (dPTE_df, rPTE_df) : tuple of pandas.DataFrame objects
        The results from pyPTE.pyPTE.PTE are stored as pandas.DataFrames, while it is indexed in two dimensions by
        pandas.DataFrame.columns of the input

    """
    time_series = data_frame.as_matrix()
    dPTE, rPTE = pyPTE.PTE(time_series)
    dPTE_df = pd.DataFrame(dPTE, index=data_frame.columns, columns=data_frame.columns)
    rPTE_df = pd.DataFrame(rPTE, index=data_frame.columns, columns=data_frame.columns)
    return dPTE_df, rPTE_df
