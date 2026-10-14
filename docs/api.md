# API reference

## Estimation

```{eval-rst}
.. autofunction:: pyPTE.PTE
.. autofunction:: pyPTE.core.pyPTE.compute_dPTE_rawPTE
.. autofunction:: pyPTE.core.pyPTE.get_bincount_auto
```

## Significance

```{eval-rst}
.. autofunction:: pyPTE.surrogate_test
.. autofunction:: pyPTE.surrogates_needed
.. autofunction:: pyPTE.group_test
.. autofunction:: pyPTE.group_contrast
.. autofunction:: pyPTE.cluster_permutation
.. autofunction:: pyPTE.benjamini_hochberg
```

## Results

```{eval-rst}
.. autoclass:: pyPTE.SignificanceResult
.. autoclass:: pyPTE.GroupResult
.. autoclass:: pyPTE.ClusterResult
```

## Adapters

These return labelled `pandas.DataFrame` results.

```{eval-rst}
.. autofunction:: pyPTE.adapters.mne_adapter.PTE_from_mne
.. autofunction:: pyPTE.adapters.mne_adapter.interpolate_mne
.. autofunction:: pyPTE.adapters.pandas_adapter.PTE_from_dataframe
.. autofunction:: pyPTE.core.handle_multi.multi_process
```
