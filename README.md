# calibration-shift

[![tests](https://github.com/real-adi1806/calibration-shift/actions/workflows/tests.yml/badge.svg)](https://github.com/real-adi1806/calibration-shift/actions/workflows/tests.yml)

**A risk model can rank patients perfectly and still give every one of them the wrong risk.**
This package measures that, and corrects the part that can be corrected with one number.

AUC only measures ranking, so it cannot see the failure at all. Move a model to a population
where the condition is more or less common, and its ranking can stay exactly the same while
its probabilities drift badly. This package reports the full calibration picture next to the
AUC, applies the standard one-number correction (the prevalence where you deploy), and tells
you honestly what that correction leaves behind.

Needs only numpy. Python 3.9 or later.

```bash
pip install .
```

---

## A real example

`examples/cimt_heldout.csv` holds 300 real predictions. They come from an age and sex model
fitted on the public China-Fundus-CIMT dataset (CC0) at a prevalence of 73.1%, then applied to
its held-out split at 50%. The file has only two columns, the outcome and the predicted risk.

```
$ calshift audit examples/cimt_heldout.csv --ci
  audit                  n=300  prevalence=0.5000  mean predicted=0.6956
    AUC                      0.894711
    observed / expected      0.7188   (1 is right; below 1 = predicts too much risk)
    calibration intercept    -1.3242   (0 is right)
    calibration slope        1.3106   (1 is right)
    ECE                      0.1995   (10 equal-count bins)
    Brier                    0.1890
    O/E 95% bootstrap CI     0.6529 to 0.7824  (2000 draws)
```

An AUC of 0.89 looks healthy. But the model predicts about 39% more risk than exists
(observed / expected 0.72), and the interval excludes 1.

**Correct it with one number**, the prevalence where it is being used:

```
$ calshift shift examples/cimt_heldout.csv --source-prevalence 0.7314 --target-prevalence 0.5 --out shifted.csv
wrote shifted.csv: 300 rows, column 'p_shifted' added (mean risk 0.6956 -> 0.5506)
ranking is unchanged, so AUC is identical; check the calibration slope on labelled data if you have any

$ calshift audit shifted.csv --prob p_shifted
  audit                  n=300  prevalence=0.5000  mean predicted=0.5506
    AUC                      0.894711
    observed / expected      0.9081   (1 is right; below 1 = predicts too much risk)
    calibration intercept    -0.3225   (0 is right)
    calibration slope        1.3106   (1 is right)
    ECE                      0.0780   (10 equal-count bins)
    Brier                    0.1370
```

The AUC is identical to six decimal places, and it has to be: the correction never changes the
order of patients. Calibration improves from 0.72 to 0.91 observed over expected. **The slope
does not move.** That residual is real: the correction assumes a case looks the same in both
populations, and here the held-out cases are older than the ones the model learned from.
`recalibrate_logistic` fixes the slope too, but it needs outcome labels from the new site.

**Ask what happens at a screening prevalence:**

```
$ calshift simulate examples/cimt_heldout.csv --source-prevalence 0.7314 --targets 0.05 0.10 0.20 0.50
  resampled to each prevalence, 2000 draws (changes the case mix only, not what a case looks like)
  target   O/E      95% CI              over-predicts   ECE      O/E after prior shift
  0.0500   0.0936   0.0885 to 0.0991    10.69 x        0.4848   0.9337
  0.1000   0.1810   0.1719 to 0.1911     5.52 x        0.4528   0.9072
  0.2000   0.3401   0.3238 to 0.3565     2.94 x        0.3884   0.8933
  0.5000   0.7191   0.6948 to 0.7447     1.39 x        0.2029   0.9089
```

At 10% prevalence, the uncorrected model over-predicts risk by 5.5 times.

---

## As a library

```python
from calshift import audit, prior_shift, recalibrate_intercept, recalibrate_logistic, at_prevalence

audit(y, p)                                   # AUC and the full calibration hierarchy, as a dict
p_new = prior_shift(p, 0.73, 0.10)            # one number, no labels needed
p_ceiling, offset = recalibrate_intercept(y, p)        # needs labels: an upper bound, not a free fix
p_full, (a, b) = recalibrate_logistic(y, p)            # needs labels: fixes the slope too
at_prevalence(y, p, 0.10, source_prevalence=0.73)      # resampled what-if
```

## What each number means

| | Perfect | Reads as |
|---|---|---|
| observed / expected | 1 | below 1: the model predicts more risk than happens |
| calibration intercept | 0 | negative: predictions too high overall |
| calibration slope | 1 | below 1: too extreme; above 1: too bunched together |
| ECE | 0 | average gap between predicted and observed, over 10 equal-count bins |

The hierarchy follows Van Calster et al., *J Clin Epidemiol* 2016;74:167-76. The one-number
correction is the prior-probability adjustment of Saerens, Latinne and Decaestecker, *Neural
Computation* 2002;14:21-41. The slope is solved exactly by Newton-Raphson to the maximum
likelihood, which matters: a general-purpose solver stopping at its default tolerance gave 1.3102
on the example above instead of 1.3106.

## What this is not

- **Not a medical device and not for decisions about patients.** It measures and adjusts model
  outputs for research and evaluation.
- **The one-number correction only fixes a change in how common the condition is.** If cases
  in the new population look different (older, more or less severe), a residual remains, and
  the slope is where it shows. Always report the slope after correcting.
- **The simulation changes the mix of cases and controls, not what a case looks like.** A real
  screening population usually has milder disease. Treat it as the prevalence part of a shift,
  isolated, not a forecast.
- **ECE with many tied predictions** (a model on two variables has lots) can depend slightly on
  the order of rows, because tied patients can fall either side of a bin edge. The slope and
  intercept don't have this problem.

## Reproducing the full analysis

This package was written alongside a calibration audit of the China-Fundus-CIMT benchmark
(paper in preparation). `examples/reproduce_paper_b.py` rebuilds the example's predictions
with that analysis's own code and checks 13 of its figures. All 13 match.

## Tests

```bash
pip install ".[dev]" && pytest -q
```

The tests generate populations where only the prevalence changes, which is the one case the
correction should fix exactly, and check that it does. A deliberately broken correction fails
them.

MIT licence.
