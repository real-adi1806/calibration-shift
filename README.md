# calibration-shift

[![tests](https://github.com/real-adi1806/calibration-shift/actions/workflows/tests.yml/badge.svg)](https://github.com/real-adi1806/calibration-shift/actions/workflows/tests.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

**A risk model can rank patients perfectly and still give every one of them the wrong risk.**
This package measures that, and corrects the part that can be corrected with one number.

On a public retinal-imaging benchmark, a model with an AUC of 0.89 predicted 39% more risk
than existed, and 5.5 times too much at a 10% screening prevalence. AUC only measures ranking,
so it cannot see this at all. `calshift` reports the full calibration picture next to the AUC,
applies the standard one-number correction (the prevalence where you deploy), and tells you
honestly what that correction leaves behind.

## Quick start

```bash
pip install git+https://github.com/real-adi1806/calibration-shift
calshift audit your_predictions.csv
```

Or run it without installing anything (needs [uv](https://docs.astral.sh/uv/)):

```bash
uvx --from git+https://github.com/real-adi1806/calibration-shift calshift audit your_predictions.csv
```

Needs only numpy. Python 3.9 or later.

## Use it on your own model

Your file needs two columns: `y`, the outcome (0 or 1), and `p`, the model's predicted risk
(0 to 1). If yours are named differently, add `--label outcome --prob risk`.

| Command | What it answers | Needs outcomes? |
|---|---|---|
| `calshift audit model.csv --ci` | Is the model calibrated here? AUC, observed / expected with a 95% interval, intercept, slope, ECE, Brier | Yes |
| `calshift shift model.csv --source-prevalence 0.73 --target-prevalence 0.10 --out new.csv` | Corrected risks for a population where the condition is more or less common | No, only the two prevalences |
| `calshift simulate model.csv --source-prevalence 0.73 --targets 0.05 0.10 0.20` | How badly would it over- or under-predict at those prevalences? | Yes |

Add `--json` to `audit` for machine-readable output. To try it on real data first, clone the
repository and run the commands below on `examples/cimt_heldout.csv`.

---

## A real example

`examples/cimt_heldout.csv` holds 300 real predictions. They come from an age and sex model
fitted on the public China-Fundus-CIMT dataset (CC0) at a prevalence of 73.1%, then applied to
its held-out split at 50%. The file has only two columns, the outcome and the predicted risk.
The dataset is described in Guo N, Fu W, Li H, et al., "High-resolution fundus images for
ophthalmomics and early cardiovascular disease prediction", *Sci Data* 2025;12:568
(https://doi.org/10.1038/s41597-025-04930-z) and deposited at
https://doi.org/10.6084/m9.figshare.27907056.

Two abbreviations used below: **O/E** is observed over expected (the share of patients who
really have the condition, divided by the average risk the model predicted), and **ECE** is
expected calibration error (the average gap between predicted and observed risk).

```
$ calshift audit examples/cimt_heldout.csv --ci
  audit                  n=300  prevalence=0.5000  mean predicted=0.6956
    AUC                      0.894711
    observed / expected      0.7188   (1 is right; below 1 = predicts too much risk)
    calibration intercept    -1.3242   (0 is right)
    calibration slope        1.3106   (1 is right)
    ECE                      0.1992   (10 bins, tied predictions kept together)
    Brier                    0.1890
    O/E 95% bootstrap CI     0.6529 to 0.7824  (2000 draws)
```

An AUC of 0.89 looks healthy. But the model predicts about 39% more risk than exists
(observed / expected 0.72), and the interval excludes 1.

**Correct it with one number**, the prevalence where it is being used:

```
$ calshift shift examples/cimt_heldout.csv --source-prevalence 0.7314 --target-prevalence 0.5 --out shifted.csv
wrote shifted.csv: 300 rows, column 'p_shifted' added (mean risk 0.6956 -> 0.5506)
ranking is unchanged, so AUC is identical. This correction can only move the calibration intercept; check the intercept and the slope on labelled data if you have any

$ calshift audit shifted.csv --prob p_shifted
  audit                  n=300  prevalence=0.5000  mean predicted=0.5506
    AUC                      0.894711
    observed / expected      0.9081   (1 is right; below 1 = predicts too much risk)
    calibration intercept    -0.3225   (0 is right)
    calibration slope        1.3106   (1 is right)
    ECE                      0.0807   (10 bins, tied predictions kept together)
    Brier                    0.1370
```

The AUC is identical to six decimal places, and it has to be: the correction multiplies every
patient's odds by the same number (computed exactly, with no clipping), so the order of patients
cannot change. Calibration improves from 0.72 to 0.91 observed over expected.

**What the correction cannot touch.** Adding the same constant to every log-odds can only move
the calibration intercept; the calibration slope is unchanged by construction (1.3106 before and
after), so the slope is not evidence about the correction either way. What matters is where the
two numbers end up. If the only difference between the populations were prevalence, and the
model was calibrated where it was fitted, the corrected intercept would sit near 0 and the slope
near 1. Here the intercept is still -0.32 after the correction and the slope is 1.31 (its 95%
interval in the accompanying audit is 1.04 to 1.73), so the held-out population differs in more
than prevalence. Its cases are older than the ones the model learned from (55.4 against 52.4
years), which plausibly explains it; that has not been tested by intervention.
`recalibrate_logistic` fixes the intercept and the slope, but it needs outcome labels from the new
site.

**Ask what happens at a screening prevalence:**

```
$ calshift simulate examples/cimt_heldout.csv --source-prevalence 0.7314 --targets 0.05 0.10 0.20 0.50
  resampled to each prevalence, 2000 draws (changes the case mix only, not what a case looks like)
  target   O/E      95% CI              over-predicts   ECE      O/E after prior shift
  0.0500   0.0936   0.0885 to 0.0991    10.69 x        0.4848   0.9337
  0.1000   0.1810   0.1719 to 0.1911     5.52 x        0.4528   0.9072
  0.2000   0.3401   0.3238 to 0.3565     2.94 x        0.3885   0.8933
  0.5000   0.7191   0.6948 to 0.7447     1.39 x        0.2037   0.9089
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
| ECE | 0 | average gap between predicted and observed, over 10 bins of about equal size; patients with the same prediction always share a bin |

The hierarchy follows Van Calster et al., *J Clin Epidemiol* 2016;74:167-76. The one-number
correction is the prior-probability adjustment of Saerens, Latinne and Decaestecker, *Neural
Computation* 2002;14:21-41. The slope is solved exactly by Newton-Raphson to the maximum
likelihood, which matters: scikit-learn's `LogisticRegression` at its default tolerance gave 1.3102
on the example above instead of 1.3106.

## What this is not

- **Not a medical device and not for decisions about patients.** It measures and adjusts model
  outputs for research and evaluation.
- **The one-number correction only fixes a change in how common the condition is.** If cases
  in the new population look different (older, more or less severe), a residual remains. It
  shows as a calibration intercept still away from 0 after correcting, and a slope away from 1
  (which the correction cannot change). Always report both after correcting.
- **The simulation changes the mix of cases and controls, not what a case looks like.** A real
  screening population usually has milder disease. Treat it as the prevalence part of a shift,
  isolated, not a forecast.
- **ECE depends on how predictions are binned.** A model on two variables gives many tied
  predictions (101 distinct values among the example's 300 patients). Equal-count bins that
  split tied patients by row order made the example's ECE range from 0.1995 to 0.2010 as the
  rows were shuffled, so calshift keeps identical predictions in one bin and moves each edge to
  the nearest tie boundary: the bins are about equal in size and the result never depends on
  row order. `expected_calibration_error(y, p, ties="split")` gives the older split binning, only
  for reproducing figures computed that way. The slope and intercept involve no binning.
- **The calibration slope can be undefined.** If one cut on the predictions separates the
  outcomes perfectly, no finite slope exists; you get `nan` and a warning rather than a huge
  meaningless number. Otherwise the fit is checked before it is returned (the score equations
  must hold), so it either gives the maximum likelihood slope or says it could not.

## Reproducing the full analysis

This package was written alongside a calibration audit of the China-Fundus-CIMT benchmark
(paper in preparation). The paper's code is a separate repository,
https://github.com/real-adi1806/fundus-cimt-calibration-audit, and you need a copy of it with
its pipeline already run (`python run_all.py`; that needs the public dataset, see its README).
The script also needs pandas, scipy and scikit-learn
(`pip install pandas scipy scikit-learn`) on top of numpy.

```bash
python examples/reproduce_paper_b.py PATH/TO/fundus-cimt-calibration-audit
```

`examples/reproduce_paper_b.py` rebuilds the example's predictions
with that analysis's own code and checks 13 of its figures: all 13 match (ranking, observed /
expected, intercept, slope, Brier and ECE). The ECE figures are checked against the paper's
stable-sort values with `ties="split"`, the binning that produced them; the paper's headline ECE
is an average over 5,000 orderings of tied predictions (see the note on ECE above). With tied
predictions kept together, the prevalence correction closes 93.3% of the ECE gap to the intercept
refit, inside the 88% to 95% range the paper reports across orderings.

## Tests

```bash
pip install ".[dev]" && pytest -q
```

The tests generate populations where only the prevalence changes, which is the one case the
correction should fix exactly, and check that it does. A deliberately broken correction fails
them.

## Citing it

If you use it in research, GitHub's "Cite this repository" button (from `CITATION.cff`) gives
the citation.

## Contributing

Bug reports and questions are welcome as issues. If you find a model or dataset where the
numbers look wrong, an issue with a small example file is the most useful thing you can send.

MIT licence.
