"""Corrections for a model moved to a population with a different case mix.

prior_shift            needs ONE number: the prevalence where you deploy. No labels.
recalibrate_intercept  needs outcome labels from the new site. An upper bound, not a free fix.
recalibrate_logistic   also refits the slope. Needs labels.
"""
import numpy as np

from .metrics import _arrays, calibration_intercept, expit, logistic_recalibration, logit


def _check_prevalence(name, value):
    if not 0 < value < 1:
        raise ValueError("%s must be strictly between 0 and 1, got %r" % (name, value))


def prior_shift(p, source_prevalence, target_prevalence):
    """Move predicted probabilities from the prevalence a model was fitted at to a new one.

    Adds logit(target) - logit(source) to every logit (Saerens, Latinne and
    Decaestecker, Neural Computation 2002). It assumes PRIOR-PROBABILITY SHIFT:
    the share of cases changes but a case looks the same in both populations.
    Where cases themselves differ (older, more severe), a residual remains that
    this cannot remove; check the calibration slope afterwards.

    The ranking of patients is unchanged, so AUC is identical before and after.
    """
    _check_prevalence("source_prevalence", source_prevalence)
    _check_prevalence("target_prevalence", target_prevalence)
    p = np.asarray(p, dtype=float)
    shift = logit(target_prevalence) - logit(source_prevalence)
    return expit(logit(p) + shift)


def recalibrate_intercept(y, p):
    """Refit only the intercept on labelled data from the new site. Returns (corrected p, offset)."""
    y, p = _arrays(y, p)
    a = calibration_intercept(y, p)
    return expit(logit(p) + a), a


def recalibrate_logistic(y, p):
    """Refit intercept and slope on labelled data from the new site. Returns (corrected p, (a, b))."""
    y, p = _arrays(y, p)
    a, b = logistic_recalibration(y, p)
    return expit(a + b * logit(p)), (a, b)
