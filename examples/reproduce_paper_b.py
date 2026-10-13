"""Reproduce the calibration audit of China-Fundus-CIMT with this package.

    python examples/reproduce_paper_b.py PATH/TO/paper-b-calibration

That repository must already have run its pipeline (it needs the public CC0
dataset in data/ and the features it builds). This script uses that repository's
own code to rebuild the age and sex model's predictions on the held-out split,
then measures and corrects them with calshift, and checks every figure against
the values the paper reports. It needs numpy, pandas and scikit-learn.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from calshift import (  # noqa: E402
    at_prevalence, audit, bootstrap_ci, observed_over_expected, prior_shift, recalibrate_intercept,
)

# What the paper reports for model A (age and sex) on the pooled held-out split.
PAPER = {
    "auc": 0.894711, "observed_over_expected": 0.718776, "calibration_intercept": -1.324221,
    # The paper prints 1.310243, which comes from scikit-learn's lbfgs stopping at its
    # default tolerance. The maximum-likelihood slope is 1.310646: lbfgs at tol 1e-12,
    # newton-cg and this package's Newton-Raphson all agree, and its log-likelihood is
    # higher (-124.04912278 against -124.04912662). Checked against the exact value.
    "calibration_slope": 1.310646, "ece": 0.199464, "brier": 0.188968,
    "oe_ci": (0.6529, 0.7824),
    "prior_shift": {"observed_over_expected": 0.9082, "ece": 0.078033},
    "refit": {"ece": 0.071022},
    "screening_0.05": 0.0936,
}


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print("  %-34s %-12.6f paper %-10.6f %s" % (name, got, want, "ok" if ok else "DIFFERENT"))
    return ok


def main(paper_b_root):
    sys.path.insert(0, os.path.join(paper_b_root, "src"))
    import calibration_audit as ca  # that repository's own module

    loaded = ca.load_real()
    if loaded is None:
        sys.exit("paper-b-calibration has not built its features yet; run its run_all.py first")
    df, _, _ = loaded
    y = df["label"].values.astype(int)
    A = df[["age", "sex"]].values.astype(np.float64)
    train = df["group"].values == 1
    p = ca.fit_predict(A[train], y[train], A[~train], C=1.0, n_cov=0)
    yh = y[~train]
    source = y[train].mean()

    print("held-out split: n=%d, prevalence %.4f; fitted at %.4f\n" % (len(yh), yh.mean(), source))
    ok = []
    r = audit(yh, p)
    for k in ("auc", "observed_over_expected", "calibration_intercept", "calibration_slope", "ece", "brier"):
        ok.append(check(k, r[k], PAPER[k], 5e-6))
    lo, hi = bootstrap_ci(yh, p, observed_over_expected, n_boot=2000, seed=42)
    ok.append(check("O/E 95% CI low", lo, PAPER["oe_ci"][0], 5e-4))
    ok.append(check("O/E 95% CI high", hi, PAPER["oe_ci"][1], 5e-4))

    shifted = prior_shift(p, source, yh.mean())
    rs = audit(yh, shifted)
    ok.append(check("prior shift: O/E", rs["observed_over_expected"], PAPER["prior_shift"]["observed_over_expected"], 5e-5))
    ok.append(check("prior shift: ECE", rs["ece"], PAPER["prior_shift"]["ece"], 5e-6))
    ok.append(check("prior shift: AUC unchanged", rs["auc"], r["auc"], 0.0))
    refit, _ = recalibrate_intercept(yh, p)
    ok.append(check("intercept refit: ECE", audit(yh, refit)["ece"], PAPER["refit"]["ece"], 5e-6))
    sim = at_prevalence(yh, p, 0.05, source_prevalence=source, draws=2000, seed=42)
    ok.append(check("resampled to 5%: O/E", sim["observed_over_expected"], PAPER["screening_0.05"], 5e-5))

    print("\n%d of %d checks match the paper" % (sum(ok), len(ok)))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
