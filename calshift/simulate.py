"""What happens to calibration at another prevalence, by resampling labelled data.

This changes the MIX of cases and controls. It cannot change what a case looks
like, so it isolates the prevalence part of a shift and nothing else. A real
screening population usually has milder disease among its cases.
"""
import numpy as np

from .correct import prior_shift
from .metrics import _arrays, expected_calibration_error


def at_prevalence(y, p, target_prevalence, source_prevalence=None, n=None, draws=2000, seed=42):
    """Resample cases and controls with replacement to a target prevalence.

    Returns the mean observed-over-expected across draws, its 95% percentile
    interval, the mean ECE, and, if source_prevalence is given, the mean
    observed-over-expected after the one-number prior shift.
    """
    y, p = _arrays(y, p)
    if not 0 < target_prevalence < 1:
        raise ValueError("target_prevalence must be strictly between 0 and 1")
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need at least one case and one control to resample")
    n = n or len(y)
    n_pos = max(2, int(round(target_prevalence * n)))
    n_neg = max(2, n - n_pos)
    rng = np.random.default_rng(seed)
    oes, eces, oes_fixed = [], [], []
    for _ in range(draws):
        idx = np.concatenate([rng.choice(pos, n_pos, replace=True), rng.choice(neg, n_neg, replace=True)])
        yy, pp = y[idx], p[idx]
        oes.append(yy.mean() / pp.mean())
        eces.append(expected_calibration_error(yy, pp))
        if source_prevalence is not None:
            oes_fixed.append(yy.mean() / prior_shift(pp, source_prevalence, target_prevalence).mean())
    out = {
        "target_prevalence": float(target_prevalence),
        "observed_over_expected": float(np.mean(oes)),
        "observed_over_expected_ci": (float(np.percentile(oes, 2.5)), float(np.percentile(oes, 97.5))),
        "overprediction_factor": float(1 / np.mean(oes)),
        "ece": float(np.mean(eces)),
    }
    if oes_fixed:
        out["observed_over_expected_after_prior_shift"] = float(np.mean(oes_fixed))
    return out
