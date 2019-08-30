import numpy as np
import pandas as pd
from scipy import stats
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

def _normal_map(original_s, std_s, prior_mean, prior_std):
    corrected_s = ((original_s * prior_std**2) + (prior_mean * std_s**2)) / \
                  (              prior_std**2  +               std_s**2)
    if isinstance(std_s, pd.Series):
        corrected_s[std_s == np.inf] = prior_mean
    elif std_s == np.inf:
        corrected_s = prior_mean
    return corrected_s

def _binom_std(sample, p):
    if sample == 0:
        return np.inf
    elif pd.isnull(sample):
        return np.inf
    return stats.binom.std(sample, p=p) / sample

def _map_corrected_pct(original_s, sample_s, overall_mean, prior_sample=50):
    if isinstance(overall_mean, pd.Series):
        std_df = pd.DataFrame({ "sample": sample_s, "overall_mean": overall_mean })
        std_s = std_df.apply(lambda row: _binom_std(row["sample"], row["overall_mean"]), axis=1)
    elif isinstance(sample_s, pd.Series):
        std_s = sample_s.apply(_binom_std, p=overall_mean)
    elif isinstance(sample_s, (int, float)):
        std_s = _binom_std(sample_s, overall_mean)
    else:
        raise ValueError()

    # Doing a poor man's maximum a posteriori estimate of a percentage using the overall sample
    # mean as the prior. https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation#Example
    #
    # Instead of the full set, we use GAUSSIAN_PRIOR_SAMPLE to model the prior's variance. To
    # intuitively interpret this, observe that the prior contributes the same amount as the sample
    # if the sample has GAUSSIAN_PRIOR_SAMPLE elements.
    prior_std = _binom_std(prior_sample, overall_mean)
    corrected_s = _normal_map(original_s, std_s, overall_mean, prior_std)

    return corrected_s, std_s
