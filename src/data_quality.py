"""
data_quality.py

Data-quality metrics for the cross-sell system (Objective 2, Req. 8b).

Two metrics are implemented:

1. Missing-value rate - a per-column completeness check with a configurable
   tolerance. Cheap, and catches the most common upstream breakage
   (a join that silently dropped values, a renamed source column).

2. Population Stability Index (PSI) - quantifies how far an incoming
   distribution has drifted from the training distribution. This is the
   metric that matters after deployment: the model does not degrade because
   the code changed, it degrades because the customers changed. Conventional
   bands are PSI < 0.10 stable, 0.10-0.25 moderate drift, > 0.25 significant
   drift and a retraining trigger.

Schema validation, the third data-quality gate, lives in preprocessing.py
next to the code that consumes it.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.config import MAX_MISSING_FRACTION, PSI_MODERATE_DRIFT, PSI_SIGNIFICANT_DRIFT
from src.logging_config import get_logger

logger = get_logger(__name__)


def missing_value_report(df: pd.DataFrame) -> Dict[str, float]:
    """Return the fraction of missing values per column and log breaches."""
    report = df.isna().mean().to_dict()

    breaches = {col: rate for col, rate in report.items() if rate > MAX_MISSING_FRACTION}
    if breaches:
        logger.warning(
            "Columns exceeding the %.0f%% missing-value tolerance: %s",
            MAX_MISSING_FRACTION * 100,
            {col: f"{rate:.2%}" for col, rate in breaches.items()},
        )
    else:
        logger.info(
            "Missing-value check passed (worst column: %.4f)",
            max(report.values()) if report else 0.0,
        )
    return report


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute PSI for one numeric column between two samples.

    The reference distribution is split into quantile bins so each bin holds
    roughly equal mass; the current sample is then bucketed with those same
    edges. Quantile bins rather than equal-width ones matter for a skewed
    column like Annual_Premium, where equal-width binning would place almost
    every record in the first bucket and hide real movement.

    Args:
        reference: the training-time distribution.
        current: the incoming/production distribution.
        bins: number of quantile bins.

    Returns:
        The PSI value. Higher means more drift.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if reference.empty or current.empty:
        logger.warning("PSI requested with an empty sample - returning 0.0")
        return 0.0

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        logger.warning("Too few distinct values in the reference sample to compute PSI")
        return 0.0

    # Open the outer edges so values beyond the training range still land in
    # a bin instead of being dropped by np.histogram.
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Clip to a small positive floor: an empty bin would otherwise make the
    # log term infinite.
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    if psi > PSI_SIGNIFICANT_DRIFT:
        logger.error(
            "PSI=%.4f exceeds the significant-drift threshold (%.2f) - retraining indicated",
            psi,
            PSI_SIGNIFICANT_DRIFT,
        )
    elif psi > PSI_MODERATE_DRIFT:
        logger.warning(
            "PSI=%.4f indicates moderate drift (threshold %.2f) - monitor closely",
            psi,
            PSI_MODERATE_DRIFT,
        )
    else:
        logger.info("PSI=%.4f - distribution stable", psi)

    return psi


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list
) -> Dict[str, float]:
    """Compute PSI for several numeric columns at once."""
    report = {}
    for column in columns:
        if column in reference.columns and column in current.columns:
            report[column] = population_stability_index(reference[column], current[column])
        else:
            logger.warning("Column %s missing from one of the samples - skipped", column)
    return report
