import numpy as np
import pandas as pd
from util.caching import cache_today

# FIXME: There are more exact significance tests with more resolving power for our
# cases, e.g. Barnard's or Boschloo's tests.
def binomial_pv(counts=None, frequencies=None, table=None, chi_min=10, **kwargs):
    if table is None:
        if not isinstance(counts, np.ndarray):
            counts = np.asarray(counts)
        if not isinstance(frequencies, np.ndarray):
            frequencies = np.asarray(frequencies)

        table = np.nan_to_num(np.vstack([
            (1 - frequencies) * counts,
            frequencies * counts,
        ]))
    if isinstance(table, pd.DataFrame):
        table = table.values

    pv = np.nan
    if table.min() >= chi_min:
        # fisher_exact slow as molasses for large Np; chi2 _should_ be a reasonable
        # approximation if Np>5 for all N, p.
        pv = stats.chi2_contingency(table)[1]

    else:
        # fisher_exact was taking 60-240s in some cases based on specific inputs.
        # it's been restricted down enough at this point that it's fast enough
        # in general, but we might be being overly cautious here.
        @cache_today
        def __fisher_exact(table):
            return stats.fisher_exact(table)[1]
        pv = __fisher_exact(table)

    return pv
