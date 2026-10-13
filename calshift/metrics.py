"""Discrimination and the calibration hierarchy, with numpy only.

Every function takes y (0/1 outcomes) and p (predicted probabilities) as
array-likes of the same length.
"""
import numpy as np

EPS = 1e-6


def _arrays(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError("y and p must have the same length, got %d and %d" % (len(y), len(p)))
    if len(y) == 0:
        raise ValueError("y and p are empty")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 and 1")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be probabilities between 0 and 1")
    return y, p


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def auc(y, p):
    """Area under the ROC curve via the rank-sum identity, ties counted as half."""
    y, p = _arrays(y, p)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # average rank for a tie group
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def observed_over_expected(y, p):
    """Event rate over mean predicted risk. 1 is right; below 1 means the model predicts too much risk."""
    y, p = _arrays(y, p)
    return float(y.mean() / p.mean()) if p.mean() > 0 else float("nan")


def calibration_intercept(y, p):
    """Calibration in the large on the logit scale: solve for a in y ~ expit(logit(p) + a).

    0 is perfect. Negative means the model predicts too high. Solved by bisection,
    which is exact to the tolerance because the left side is monotone in a.
    """
    y, p = _arrays(y, p)
    lo = logit(p)
    target = y.sum()
    if target == 0 or target == len(y):
        return float("nan")
    a_lo, a_hi = -25.0, 25.0
    for _ in range(200):
        mid = (a_lo + a_hi) / 2.0
        if expit(lo + mid).sum() < target:
            a_lo = mid
        else:
            a_hi = mid
    return float((a_lo + a_hi) / 2.0)


def logistic_recalibration(y, p, max_iter=100, tol=1e-12):
    """Fit y ~ expit(a + b * logit(p)) by Newton-Raphson. Returns (a, b).

    b is the calibration slope: 1 is perfect, below 1 means predictions are too
    extreme, above 1 means they are too bunched together.
    """
    y, p = _arrays(y, p)
    if y.sum() in (0, len(y)):
        return float("nan"), float("nan")
    X = np.column_stack([np.ones(len(y)), logit(p)])
    beta = np.array([0.0, 1.0])
    for _ in range(max_iter):
        mu = expit(X @ beta)
        w = mu * (1 - mu)
        grad = X.T @ (y - mu)
        hess = (X * w[:, None]).T @ X
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return float(beta[0]), float(beta[1])


def calibration_slope(y, p):
    return logistic_recalibration(y, p)[1]


def expected_calibration_error(y, p, bins=10):
    """ECE over equal-count bins, ordered by predicted risk.

    With heavily tied predictions (a model on two variables has many), patients
    with the same prediction can fall either side of a bin edge, so ECE then
    depends slightly on the order of the input rows. That is a property of
    equal-count ECE, not of this code; report the slope and intercept alongside.
    """
    y, p = _arrays(y, p)
    n = len(y)
    if n < bins * 2:
        bins = max(2, n // 5)
    order = np.argsort(p)
    y, p = y[order], p[order]
    edges = np.linspace(0, n, bins + 1).astype(int)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            total += (b - a) / n * abs(y[a:b].mean() - p[a:b].mean())
    return float(total)


def brier(y, p):
    y, p = _arrays(y, p)
    return float(np.mean((p - y) ** 2))


def audit(y, p):
    """The full row: discrimination, then every level of the calibration hierarchy."""
    y, p = _arrays(y, p)
    a, b = logistic_recalibration(y, p)
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "auc": auc(y, p),
        "observed_over_expected": observed_over_expected(y, p),
        "calibration_intercept": calibration_intercept(y, p),
        "calibration_slope": b,
        "ece": expected_calibration_error(y, p),
        "brier": brier(y, p),
    }


def bootstrap_ci(y, p, fn, n_boot=2000, seed=42, level=0.95):
    """Percentile bootstrap interval for any scalar fn(y, p). Resamples rows with replacement."""
    y, p = _arrays(y, p)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        v = fn(yy, p[idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return float("nan"), float("nan")
    tail = (1 - level) / 2 * 100
    return float(np.percentile(vals, tail)), float(np.percentile(vals, 100 - tail))
