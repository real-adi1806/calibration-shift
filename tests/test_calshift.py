import csv
import warnings

import numpy as np
import pytest

from calshift import (
    at_prevalence,
    auc,
    audit,
    calibration_intercept,
    calibration_slope,
    expected_calibration_error,
    observed_over_expected,
    prior_shift,
    recalibrate_intercept,
    recalibrate_logistic,
)
from calshift.cli import main
from calshift.metrics import expit, logit


def population(prevalence, n, seed, sep=1.5):
    """Cases and controls drawn from FIXED distributions; only the share of cases changes.

    This is pure prior-probability shift, the one case the correction is meant
    to fix exactly. The returned p is the true probability for a population at
    `fitted_at` prevalence, i.e. a perfectly calibrated model for that prevalence.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(float)
    x = np.where(y == 1, rng.normal(sep / 2, 1, n), rng.normal(-sep / 2, 1, n))
    log_lr = sep * x                       # log likelihood ratio for two unit-variance normals
    return y, log_lr


def model_fitted_at(log_lr, prevalence):
    return expit(log_lr + logit(prevalence))


# ── discrimination ───────────────────────────────────────────────────────────

def test_auc_hand_example():
    assert auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_auc_counts_ties_as_half():
    assert auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_prior_shift_leaves_auc_exactly_unchanged():
    y, llr = population(0.7, 5000, seed=1)
    p = model_fitted_at(llr, 0.7)
    assert auc(y, prior_shift(p, 0.7, 0.1)) == auc(y, p)


# ── the correction does what it claims, and only that ─────────────────────────

def test_prior_shift_repairs_a_pure_prevalence_shift():
    _, llr_src = population(0.7, 20000, seed=2)
    y_tgt, llr_tgt = population(0.1, 20000, seed=3)
    p = model_fitted_at(llr_tgt, 0.7)                 # model fitted at 70%, applied at 10%
    assert observed_over_expected(y_tgt, p) < 0.4     # badly over-predicts before correction
    fixed = prior_shift(p, 0.7, 0.1)
    assert observed_over_expected(y_tgt, fixed) == pytest.approx(1.0, abs=0.05)
    assert calibration_slope(y_tgt, fixed) == pytest.approx(1.0, abs=0.1)


def test_prior_shift_is_a_no_op_at_the_same_prevalence():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(prior_shift(p, 0.3, 0.3), p, atol=1e-12)


def test_intercept_refit_reaches_the_ceiling():
    y, llr = population(0.2, 4000, seed=4)
    p = model_fitted_at(llr, 0.6)
    fixed, offset = recalibrate_intercept(y, p)
    assert offset < 0
    assert observed_over_expected(y, fixed) == pytest.approx(1.0, abs=1e-6)
    assert calibration_intercept(y, fixed) == pytest.approx(0.0, abs=1e-6)


def test_logistic_recalibration_fixes_the_slope_too():
    y, llr = population(0.3, 8000, seed=5)
    bunched = expit(0.5 * (llr + logit(0.3)))         # logits squeezed towards 0
    fixed, (a, b) = recalibrate_logistic(y, bunched)
    assert b == pytest.approx(2.0, abs=0.15)
    assert calibration_slope(y, fixed) == pytest.approx(1.0, abs=1e-6)


# ── the metrics themselves ───────────────────────────────────────────────────

def test_slope_is_one_on_a_calibrated_model():
    y, llr = population(0.3, 20000, seed=6)
    assert calibration_slope(y, model_fitted_at(llr, 0.3)) == pytest.approx(1.0, abs=0.08)


def test_slope_above_one_means_bunched_predictions():
    y, llr = population(0.3, 20000, seed=7)
    assert calibration_slope(y, expit(0.5 * (llr + logit(0.3)))) > 1.5


def _shifted_case(seed):
    """Predictions far from 0.5 with a weaker true slope: where plain Newton from (0, 1) blew up."""
    r = np.random.default_rng(seed)
    n = int(r.integers(80, 300))
    z = r.normal(r.uniform(-4, 4), r.uniform(0.3, 2.5), n)
    a, b = r.uniform(-3, 3), r.uniform(0.1, 1.5)
    y = (r.random(n) < expit(a + b * z)).astype(float)
    return y, expit(z)


def _is_the_maximum(y, p, a, b):
    """The definition of the fit, checked directly: the score equations hold and no
    nearby (a, b) has a higher log-likelihood."""
    x = logit(p)
    resid = y - expit(a + b * x)
    if max(abs(resid.sum()), abs((x * resid).sum())) > 1e-6 * len(y):
        return False

    def ll(a_, b_):
        z = a_ + b_ * x
        return np.sum(y * z - np.logaddexp(0, z))

    best = ll(a, b)
    return all(ll(a + da, b + db) <= best + 1e-9
               for da in (-1e-3, 0, 1e-3) for db in (-1e-3, 0, 1e-3))


def test_slope_fit_reaches_the_maximum_where_plain_newton_diverged():
    # Seed 2 is one of the 93 in 400 cases on which the earlier plain Newton raised
    # "Singular matrix" or returned a slope of about 1e58 (found 25 Oct 2026).
    y, p = _shifted_case(2)
    a, b = recalibrate_logistic(y, p)[1]
    assert np.isfinite(b) and _is_the_maximum(y, p, a, b)


def test_slope_fit_is_the_maximum_on_many_random_datasets():
    for seed in range(150):
        y, p = _shifted_case(seed)
        if y.sum() in (0, len(y)):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a, b = recalibrate_logistic(y, p)[1]
        if np.isnan(b):
            x = logit(p)   # nan is only allowed when the outcomes are separated
            assert x[y == 0].max() <= x[y == 1].min() or x[y == 1].max() <= x[y == 0].min(), seed
            continue
        assert _is_the_maximum(y, p, a, b), seed


def test_slope_is_nan_with_a_warning_when_outcomes_are_separated():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    with pytest.warns(RuntimeWarning, match="separated"):
        assert np.isnan(calibration_slope(y, p))


def test_ece_hand_example():
    # two equal-count bins: {0.1, 0.2} with outcomes {0, 0}, {0.8, 0.9} with {1, 0}
    y = [0, 0, 1, 0]
    p = [0.1, 0.2, 0.8, 0.9]
    expected = 0.5 * abs(0 - 0.15) + 0.5 * abs(0.5 - 0.85)
    assert expected_calibration_error(y, p, bins=2) == pytest.approx(expected)


def test_ece_ties_keep_input_order_so_every_machine_agrees():
    # 1,000 rows, only 7 distinct predictions, so ties straddle every bin edge.
    # Small arrays would not test this: numpy sorts them stably whatever it is
    # asked, so the array has to be long enough for an unstable sort to reorder.
    rng = np.random.default_rng(3)
    p = rng.choice([0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9], size=1000)
    y = (rng.random(1000) < p).astype(int)
    # Reference: Python's sorted() is guaranteed stable, so tied rows keep their order.
    order = sorted(range(1000), key=lambda i: p[i])
    ys, ps = y[order], p[order]
    edges = np.linspace(0, 1000, 11).astype(int)
    expected = sum((b - a) / 1000 * abs(ys[a:b].mean() - ps[a:b].mean())
                   for a, b in zip(edges[:-1], edges[1:]))
    assert expected_calibration_error(y, p) == pytest.approx(expected, abs=1e-12)


def test_audit_returns_every_level():
    y, llr = population(0.4, 2000, seed=8)
    r = audit(y, model_fitted_at(llr, 0.4))
    for key in ("auc", "observed_over_expected", "calibration_intercept", "calibration_slope", "ece", "brier"):
        assert np.isfinite(r[key])


@pytest.mark.parametrize("y, p", [([0, 1], [0.5]), ([0, 2], [0.1, 0.2]), ([0, 1], [0.1, 1.5]), ([], [])])
def test_bad_input_is_refused(y, p):
    with pytest.raises(ValueError):
        audit(y, p)


def test_prevalence_must_be_a_proportion():
    with pytest.raises(ValueError):
        prior_shift([0.5], 0.7, 1.0)


# ── simulation ───────────────────────────────────────────────────────────────

def test_simulation_control_row_is_a_no_op():
    y, llr = population(0.5, 600, seed=9)
    p = model_fitted_at(llr, 0.7)
    r = at_prevalence(y, p, 0.7, source_prevalence=0.7, draws=200)
    assert r["observed_over_expected_after_prior_shift"] == pytest.approx(r["observed_over_expected"], rel=1e-9)


def test_simulation_shows_overprediction_rising_as_prevalence_falls():
    y, llr = population(0.5, 600, seed=10)
    p = model_fitted_at(llr, 0.7)
    oe = [at_prevalence(y, p, t, draws=200)["observed_over_expected"] for t in (0.5, 0.2, 0.05)]
    assert oe[0] > oe[1] > oe[2]


# ── command line ─────────────────────────────────────────────────────────────

def test_cli_audit_and_shift(tmp_path, capsys):
    y, llr = population(0.3, 400, seed=11)
    p = model_fitted_at(llr, 0.7)
    path = tmp_path / "preds.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y", "p"])
        w.writerows(zip(y.astype(int), p))
    assert main(["audit", str(path)]) == 0
    assert "observed / expected" in capsys.readouterr().out
    out = tmp_path / "shifted.csv"
    assert main(["shift", str(path), "--source-prevalence", "0.7", "--target-prevalence", "0.3", "--out", str(out)]) == 0
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 400 and "p_shifted" in rows[0]
