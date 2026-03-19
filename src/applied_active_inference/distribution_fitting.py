"""Distribution fitting for daily sales data.

Since the dataset provides one Avg_Daily_Sales value per SKU (not a time series),
we group SKUs by product category and fit parametric distributions to the
collection of average daily sales values within each group.  This captures
the distribution of "typical daily sales for a product in this category".

Candidate distributions are fit via MLE, then ranked by AIC.  The best fit
is used downstream to:
  1. Sample synthetic daily sales in the simulation.
  2. Compute 5-sigma stock levels (mean + 5*std).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rv_continuous


# ── Candidate distributions ─────────────────────────────────────────────────
# These are all continuous, non-negative distributions suitable for
# modelling daily sales quantities.
CANDIDATE_DISTRIBUTIONS: list[rv_continuous] = [
    stats.norm,         # symmetric baseline
    stats.lognorm,      # right-skewed, common for sales data
    stats.gamma,        # flexible shape, positive support
    stats.expon,        # memoryless decay — simple baseline
    stats.weibull_min,  # flexible shape parameter
]


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class FitResult:
    """Result of fitting a single distribution to observed data."""

    distribution_name: str          # scipy name, e.g. "lognorm"
    params: tuple                   # MLE parameters (shape, loc, scale)
    ks_statistic: float             # Kolmogorov–Smirnov test statistic
    ks_pvalue: float                # KS test p-value (higher = better fit)
    aic: float                      # Akaike Information Criterion
    bic: float                      # Bayesian Information Criterion
    mean: float                     # distribution mean
    std: float                      # distribution standard deviation
    stock_level_5sigma: float       # mean + 5 * std


# ── Internal helpers ─────────────────────────────────────────────────────────

def _compute_aic_bic(
    data: np.ndarray,
    dist: rv_continuous,
    params: tuple,
) -> tuple[float, float]:
    """Compute AIC and BIC for a fitted distribution.

    AIC = 2k - 2 ln(L)
    BIC = k ln(n) - 2 ln(L)

    where k = number of estimated parameters, n = sample size,
    L = likelihood evaluated at the MLE parameters.
    """
    log_likelihood = np.sum(dist.logpdf(data, *params))
    k = len(params)  # number of estimated parameters
    n = len(data)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return aic, bic


# ── Public API ───────────────────────────────────────────────────────────────

def fit_distribution(data: np.ndarray, dist: rv_continuous) -> FitResult:
    """Fit a single scipy distribution to *data* via MLE.

    Parameters
    ----------
    data : 1-D array of observed values.
    dist : a ``scipy.stats`` continuous distribution object.

    Returns
    -------
    FitResult with goodness-of-fit metrics and derived statistics.
    """
    # Maximum-likelihood estimation of parameters
    params = dist.fit(data)

    # Kolmogorov–Smirnov goodness-of-fit test
    ks_stat, ks_pvalue = stats.kstest(data, dist.cdf, args=params)

    # Information criteria
    aic, bic = _compute_aic_bic(data, dist, params)

    # Moments derived from the fitted distribution
    mean = float(dist.mean(*params))
    std = float(dist.std(*params))

    return FitResult(
        distribution_name=dist.name,
        params=params,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pvalue,
        aic=aic,
        bic=bic,
        mean=mean,
        std=std,
        stock_level_5sigma=mean + 5.0 * std,
    )


def fit_best_distribution(
    data: np.ndarray,
    candidates: list[rv_continuous] | None = None,
) -> FitResult:
    """Try multiple distributions and return the best fit (lowest AIC).

    Parameters
    ----------
    data : 1-D array of observed values (e.g. Avg_Daily_Sales for a category).
    candidates : distributions to try.  Defaults to ``CANDIDATE_DISTRIBUTIONS``.

    Returns
    -------
    FitResult for the best-fitting distribution.

    Raises
    ------
    RuntimeError
        If none of the candidate distributions could be fit.
    """
    if candidates is None:
        candidates = CANDIDATE_DISTRIBUTIONS

    # Drop non-finite values
    data = data[np.isfinite(data)]

    results: list[FitResult] = []
    for dist in candidates:
        try:
            result = fit_distribution(data, dist)
            # Sanity-check: skip if log-likelihood blew up
            if np.isfinite(result.aic):
                results.append(result)
        except Exception:
            # Distribution may fail to fit (e.g. gamma on data with zeros)
            continue

    if not results:
        raise RuntimeError("No candidate distribution could be fit to the data.")

    # Select the distribution with the lowest AIC
    results.sort(key=lambda r: r.aic)
    return results[0]


def fit_categories(
    df: pd.DataFrame,
    group_col: str = "Category",
    value_col: str = "Avg_Daily_Sales",
    min_samples: int = 10,
) -> dict[str, FitResult]:
    """Fit the best distribution for each product-category group.

    Parameters
    ----------
    df : cleaned grocery DataFrame (output of ``load_grocery_data``).
    group_col : column to group by (default ``"Category"``).
    value_col : column containing the values to fit (default ``"Avg_Daily_Sales"``).
    min_samples : skip groups with fewer than this many observations.

    Returns
    -------
    dict mapping group name -> ``FitResult``.
    """
    results: dict[str, FitResult] = {}
    for name, group in df.groupby(group_col):
        data = group[value_col].dropna().values
        if len(data) < min_samples:
            continue
        results[str(name)] = fit_best_distribution(data)
    return results
