"""Corrections for a model moved to a population with a different case mix.

prior_shift            needs ONE number: the prevalence where you deploy. No labels.
recalibrate_intercept  needs outcome labels from the new site. An upper bound, not a free fix.
recalibrate_logistic   also refits the slope. Needs labels.
"""
import numpy as np

from .metrics import _arrays, _probabilities, _scale_odds, calibration_intercept, logistic_recalibration


def _check_prevalence(name, value):
    if not 0 < value < 1:
        raise ValueError("%s must be strictly between 0 and 1, got %r" % (name, value))


def _odds(prevalence):
    return prevalence / (1 - prevalence)


def prior_shift(p, source_prevalence, target_prevalence):
    """Move predicted probabilities from the prevalence a model was fitted at to a new one.

    Adds logit(target) - logit(source) to every logit (Saerens, Latinne and
    Decaestecker, Neural Computation 2002). It assumes PRIOR-PROBABILITY SHIFT:
    the share of cases changes but a case looks the same in both populations.

    Because it adds the same constant to every logit, it can only move the
    calibration intercept: the calibration slope is unchanged by construction.
    Where cases themselves differ (older, more severe), a residual remains that
    this cannot remove; check the intercept and the slope on labelled data.

    Computed exactly in odds, without clipping, so the ranking of patients is
    unchanged and AUC is identical before and after, even for predictions of
    exactly 0 or 1.
    """
    _check_prevalence("source_prevalence", source_prevalence)
    _check_prevalence("target_prevalence", target_prevalence)
    p = _probabilities(p)
    return _scale_odds(p, _odds(target_prevalence) / _odds(source_prevalence))


def recalibrate_intercept(y, p):
    """Refit only the intercept on labelled data from the new site. Returns (corrected p, offset)."""
    y, p = _arrays(y, p)
    a = calibration_intercept(y, p)
    return _scale_odds(p, np.exp(a)), a


def recalibrate_logistic(y, p):
    """Refit intercept and slope on labelled data from the new site. Returns (corrected p, (a, b))."""
    y, p = _arrays(y, p)
    a, b = logistic_recalibration(y, p)
    # expit(a + b * logit(p)), written in odds so that it needs no clipping:
    # 0 and 1 stay 0 and 1, and distinct predictions stay distinct.
    with np.errstate(divide="ignore", over="ignore"):
        return 1.0 / (1.0 + np.exp(-a) * ((1 - p) / p) ** b), (a, b)
