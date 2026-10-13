"""Command line: calshift audit | shift | simulate, reading a CSV of outcomes and predictions."""
import argparse
import csv
import json
import sys

import numpy as np

from .correct import prior_shift
from .metrics import audit, bootstrap_ci, observed_over_expected
from .simulate import at_prevalence


def _read(path, prob_col, label_col=None):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("error: %s has no rows" % path)
    for col in [c for c in (prob_col, label_col) if c]:
        if col not in rows[0]:
            raise SystemExit("error: column %r not in %s (columns: %s)" % (col, path, ", ".join(rows[0])))
    p = np.array([float(r[prob_col]) for r in rows])
    y = np.array([float(r[label_col]) for r in rows]) if label_col else None
    return rows, y, p


def _print_audit(name, r):
    print("  %-22s n=%d  prevalence=%.4f  mean predicted=%.4f" % (name, r["n"], r["prevalence"], r["mean_predicted"]))
    print("    AUC                      %.6f" % r["auc"])
    print("    observed / expected      %.4f   (1 is right; below 1 = predicts too much risk)" % r["observed_over_expected"])
    print("    calibration intercept    %.4f   (0 is right)" % r["calibration_intercept"])
    print("    calibration slope        %.4f   (1 is right)" % r["calibration_slope"])
    print("    ECE                      %.4f   (10 equal-count bins)" % r["ece"])
    print("    Brier                    %.4f" % r["brier"])


def cmd_audit(a):
    _, y, p = _read(a.csv, a.prob, a.label)
    r = audit(y, p)
    if a.ci:
        r["observed_over_expected_ci"] = bootstrap_ci(y, p, observed_over_expected, n_boot=a.boot, seed=a.seed)
    if a.json:
        print(json.dumps(r, indent=2))
        return 0
    _print_audit("audit", r)
    if a.ci:
        lo, hi = r["observed_over_expected_ci"]
        print("    O/E 95%% bootstrap CI     %.4f to %.4f  (%d draws)" % (lo, hi, a.boot))
    return 0


def cmd_shift(a):
    rows, _, p = _read(a.csv, a.prob)
    q = prior_shift(p, a.source_prevalence, a.target_prevalence)
    out_col = a.prob + "_shifted"
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + [out_col])
        w.writeheader()
        for r, v in zip(rows, q):
            w.writerow({**r, out_col: "%.10f" % v})
    print("wrote %s: %d rows, column %r added (mean risk %.4f -> %.4f)" % (a.out, len(rows), out_col, p.mean(), q.mean()))
    print("ranking is unchanged, so AUC is identical; check the calibration slope on labelled data if you have any")
    return 0


def cmd_simulate(a):
    _, y, p = _read(a.csv, a.prob, a.label)
    results = [at_prevalence(y, p, t, source_prevalence=a.source_prevalence, draws=a.draws, seed=a.seed)
               for t in a.targets]
    if a.json:
        print(json.dumps(results, indent=2))
        return 0
    print("  resampled to each prevalence, %d draws (changes the case mix only, not what a case looks like)" % a.draws)
    head = "  target   O/E      95% CI              over-predicts   ECE"
    if a.source_prevalence is not None:
        head += "      O/E after prior shift"
    print(head)
    for r in results:
        lo, hi = r["observed_over_expected_ci"]
        line = "  %.4f   %.4f   %.4f to %.4f   %6.2f x        %.4f" % (
            r["target_prevalence"], r["observed_over_expected"], lo, hi, r["overprediction_factor"], r["ece"])
        if "observed_over_expected_after_prior_shift" in r:
            line += "   %.4f" % r["observed_over_expected_after_prior_shift"]
        print(line)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="calshift", description="Check and correct a risk model's calibration when the case mix changes.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("audit", help="AUC plus the full calibration hierarchy")
    s.add_argument("csv")
    s.add_argument("--label", default="y")
    s.add_argument("--prob", default="p")
    s.add_argument("--ci", action="store_true", help="bootstrap interval for observed/expected")
    s.add_argument("--boot", type=int, default=2000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("shift", help="apply the one-number prior shift to a column of predictions")
    s.add_argument("csv")
    s.add_argument("--prob", default="p")
    s.add_argument("--source-prevalence", type=float, required=True, help="prevalence the model was fitted at")
    s.add_argument("--target-prevalence", type=float, required=True, help="prevalence where you deploy it")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_shift)

    s = sub.add_parser("simulate", help="resample labelled data to other prevalences and measure calibration")
    s.add_argument("csv")
    s.add_argument("--label", default="y")
    s.add_argument("--prob", default="p")
    s.add_argument("--targets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30, 0.50])
    s.add_argument("--source-prevalence", type=float, default=None)
    s.add_argument("--draws", type=int, default=2000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_simulate)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except ValueError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
